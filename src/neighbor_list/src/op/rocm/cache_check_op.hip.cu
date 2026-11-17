#include "cache_check_op.h"

#include "common/rbmd_define.h"

namespace op {
namespace {

__device__ void AtomicMaxNonNegative(rbmd::Real* destination,
                                     rbmd::Real value) {
#if USE_DOUBLE
  atomicMax(reinterpret_cast<unsigned long long*>(destination),
            static_cast<unsigned long long>(__double_as_longlong(value)));
#else
  atomicMax(reinterpret_cast<unsigned int*>(destination),
            __float_as_uint(value));
#endif
}

constexpr int kCacheCheckItemsPerThread = 4;

struct MaxReal {
  __device__ __forceinline__ rbmd::Real operator()(rbmd::Real lhs,
                                                    rbmd::Real rhs) const {
    return lhs > rhs ? lhs : rhs;
  }
};

template <bool collect_max_displacement>
__global__ void CacheCheckKernel(
    const rbmd::Id* __restrict__ atom_ids,
    const rbmd::Id* __restrict__ cached_atom_ids,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Real* __restrict__ cached_px,
    const rbmd::Real* __restrict__ cached_py,
    const rbmd::Real* __restrict__ cached_pz, Box box, rbmd::Id atom_count,
    rbmd::Real rebuild_threshold_sq, rbmd::Real* __restrict__ status) {
  const rbmd::Id block_base =
      blockIdx.x * blockDim.x * kCacheCheckItemsPerThread;
  rbmd::Real thread_max_displacement_sq = rbmd::Real(0);
#pragma unroll
  for (int item = 0; item < kCacheCheckItemsPerThread; ++item) {
    const rbmd::Id atom =
        block_base + threadIdx.x + item * blockDim.x;
    if (atom >= atom_count) {
      continue;
    }
    const bool atom_order_changed =
        __ldg(&atom_ids[atom]) != __ldg(&cached_atom_ids[atom]);
    rbmd::Real dx = __ldg(&px[atom]) - __ldg(&cached_px[atom]);
    rbmd::Real dy = __ldg(&py[atom]) - __ldg(&cached_py[atom]);
    rbmd::Real dz = __ldg(&pz[atom]) - __ldg(&cached_pz[atom]);
    MinImageDistance_while(box, dx, dy, dz);
    const rbmd::Real displacement_sq = dx * dx + dy * dy + dz * dz;

    // Match the HOOMD/LAMMPS rebuild check: the common reuse path performs no
    // global atomic operation. Only a particle that invalidates the cache sets
    // the flag. Exact maximum displacement is retained for explicit diagnostics.
    if (atom_order_changed || displacement_sq > rebuild_threshold_sq) {
      AtomicMaxNonNegative(&status[0], rbmd::Real(1));
    }
    if constexpr (collect_max_displacement) {
      thread_max_displacement_sq =
          MaxReal()(thread_max_displacement_sq, displacement_sq);
    }
  }
  if constexpr (collect_max_displacement) {
    using BlockReduce = BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const rbmd::Real block_max =
        BlockReduce(temp_storage).Reduce(thread_max_displacement_sq, MaxReal());
    if (threadIdx.x == 0) {
      AtomicMaxNonNegative(&status[1], block_max);
    }
  }
}

}  // namespace

void CacheCheckOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id* atom_ids, const rbmd::Id* cached_atom_ids,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* cached_px, const rbmd::Real* cached_py,
    const rbmd::Real* cached_pz, Box box, rbmd::Id atom_count,
    rbmd::Real rebuild_threshold_sq, bool collect_max_displacement,
    rbmd::Real* status) {
  if (atom_count <= 0) {
    return;
  }
  constexpr unsigned int atoms_per_block =
      BLOCK_SIZE * kCacheCheckItemsPerThread;
  const unsigned int blocks = static_cast<unsigned int>(
      (atom_count + atoms_per_block - 1) / atoms_per_block);
  if (collect_max_displacement) {
    CHECK_KERNEL(CacheCheckKernel<true><<<blocks, BLOCK_SIZE, 0, 0>>>(
        atom_ids, cached_atom_ids, px, py, pz, cached_px, cached_py, cached_pz,
        box, atom_count, rebuild_threshold_sq, status));
  } else {
    CHECK_KERNEL(CacheCheckKernel<false><<<blocks, BLOCK_SIZE, 0, 0>>>(
        atom_ids, cached_atom_ids, px, py, pz, cached_px, cached_py, cached_pz,
        box, atom_count, rebuild_threshold_sq, status));
  }
}

}  // namespace op
