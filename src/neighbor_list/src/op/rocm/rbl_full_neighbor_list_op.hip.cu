#include "full_neighbor_list_op.h"
#include "common/atomic_count.h"
// #include "hiprand/hiprand_kernel.h" TODO
#include "rbl_full_neighbor_list_op.h"
#include <cstdint>
#include <cstdlib>
#include <limits>
#include "common/rbl_fixedp_uniform.hpp"
namespace op {

__device__ unsigned long long MixRblSampleKey(unsigned long long value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ bool RblShellSampleSelected(rbmd::Id selection_frequency,
                                       rbmd::Id sample_epoch,
                                       rbmd::Id atom_id,
                                       rbmd::Id neighbor_id) {
  if (selection_frequency <= 1) {
    return true;
  }
  unsigned long long key =
      static_cast<unsigned long long>(static_cast<long long>(sample_epoch));
  key ^= MixRblSampleKey(
      static_cast<unsigned long long>(static_cast<long long>(atom_id)));
  key ^= MixRblSampleKey(
      static_cast<unsigned long long>(static_cast<long long>(neighbor_id)) +
      0xd1b54a32d192ed03ULL);
  return MixRblSampleKey(key) %
             static_cast<unsigned long long>(selection_frequency) ==
         0ULL;
}

// TODO warp op and optimize
//! Note: the warp implementation still needs dedicated validation.
__global__ void GenerateRBLFullNeighborList(
    rbmd::Id* __restrict__ per_atom_cell_id,
    rbmd::Id* __restrict__ in_atom_list_start_index,
    rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real trunc_distance_power_2, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* __restrict__ atoms_id,
    rbmd::Real* __restrict__ px,
    rbmd::Real* __restrict__ py, rbmd::Real* __restrict__ pz,
    rbmd::Id* __restrict__ sorted_atom_indices,
    const rbmd::Id* __restrict__ neighbor_cell,
    rbmd::Id neighbor_cell_num,
    rbmd::Id* __restrict__ actual_neighbor_num,
    rbmd::Id max_neighbor_num,
    rbmd::Id* __restrict__ neighbor_start, rbmd::Id* __restrict__ neighbor_end,
    rbmd::Id* __restrict__ neighbors, rbmd::Id random_neighbor_capacity,
    rbmd::Id* __restrict__ random_neighbors,
    rbmd::Id* __restrict__ random_neighbors_num,
    rbmd::Real* __restrict__ random_neighbor_weight,
    rbmd::Id* __restrict__ required_random_neighbor_capacity, Box box,
    rbmd::Id* __restrict__ should_realloc,
    rbmd::Id* __restrict__ required_neighbor_capacity,
    rbmd::Id selection_frequency, rbmd::Id sample_epoch) {
  const unsigned int atom_idx = (blockIdx.x * blockDim.x + threadIdx.x);
  if (atom_idx < total_atom_num) {
    rbmd::Id neighbor_num = neighbor_start[atom_idx];
    rbmd::Id sampled_neighbor_num = 0;
    rbmd::Id total_neighbor_num = 0;
    rbmd::Id cell_idx = __ldg(&per_atom_cell_id[atom_idx]);
    const std::uint64_t neighbor_cell_row =
        static_cast<std::uint64_t>(cell_idx) *
        static_cast<std::uint64_t>(neighbor_cell_num);
    const rbmd::Real center_x = __ldg(&px[atom_idx]);
    const rbmd::Real center_y = __ldg(&py[atom_idx]);
    const rbmd::Real center_z = __ldg(&pz[atom_idx]);
    for (rbmd::Id i = 0; i < neighbor_cell_num; ++i) {
      const rbmd::Id neighbour_cell_idx = __ldg(
          &neighbor_cell[neighbor_cell_row + static_cast<std::uint64_t>(i)]);
      rbmd::Id start = __ldg(&in_atom_list_start_index[neighbour_cell_idx]);
      rbmd::Id end = __ldg(&in_atom_list_end_index[neighbour_cell_idx]);
      for (rbmd::Id neighbor_atom_pos = start; neighbor_atom_pos < end;
           ++neighbor_atom_pos) {
        const rbmd::Id neighbor_atom_idx =
            __ldg(&sorted_atom_indices[neighbor_atom_pos]);
        if (atom_idx != neighbor_atom_idx) {
          const rbmd::Real distance = CaculateDistance(
              box, center_x, center_y, center_z,
              __ldg(&px[neighbor_atom_idx]),
              __ldg(&py[neighbor_atom_idx]), __ldg(&pz[neighbor_atom_idx]));
          if (distance < trunc_distance_power_2) {
            if (total_neighbor_num < max_neighbor_num) {
              neighbors[neighbor_num] = neighbor_atom_idx;
              ++neighbor_num;
            } else {
              rbmd::AtomicOrCount(should_realloc, RBMD_TRUE);
            }
            ++total_neighbor_num;
          } else if (distance < cutoff_2 &&
                     RblShellSampleSelected(
                         selection_frequency, sample_epoch,
                         __ldg(&atoms_id[atom_idx]),
                         __ldg(&atoms_id[neighbor_atom_idx]))) {
            if (sampled_neighbor_num < random_neighbor_capacity) {
              const unsigned long output =
                  static_cast<unsigned long>(sampled_neighbor_num) *
                      static_cast<unsigned long>(total_atom_num) +
                  static_cast<unsigned long>(atom_idx);
              random_neighbors[output] = neighbor_atom_idx;
            }
            ++sampled_neighbor_num;
          }
        }
      }
    }
    neighbor_end[atom_idx] = neighbor_num;
    rbmd::AtomicMaxCount(required_neighbor_capacity, total_neighbor_num);
    actual_neighbor_num[atom_idx] = total_neighbor_num;
    random_neighbors_num[atom_idx] =
        MIN(sampled_neighbor_num, random_neighbor_capacity);
    random_neighbor_weight[atom_idx] = static_cast<rbmd::Real>(
        selection_frequency > 0 ? selection_frequency : rbmd::Id(1));
    rbmd::AtomicMaxCount(required_random_neighbor_capacity, sampled_neighbor_num);
    rbmd::Id my_total_neighbor_num = total_neighbor_num;
    if (my_total_neighbor_num > max_neighbor_num) {
      rbmd::AtomicOrCount(should_realloc, RBMD_TRUE);
    }
  }
}

template <int THREADS_PER_PARTICLE>
__global__ void GenerateRBLFullNeighborListGrouped(
    rbmd::Id* __restrict__ per_atom_cell_id,
    rbmd::Id* __restrict__ in_atom_list_start_index,
    rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real trunc_distance_power_2, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* __restrict__ atoms_id,
    rbmd::Real* __restrict__ px, rbmd::Real* __restrict__ py,
    rbmd::Real* __restrict__ pz,
    rbmd::Id* __restrict__ sorted_atom_indices,
    const rbmd::Id* __restrict__ neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id* __restrict__ actual_neighbor_num,
    rbmd::Id max_neighbor_num, rbmd::Id* __restrict__ neighbor_start,
    rbmd::Id* __restrict__ neighbor_end, rbmd::Id* __restrict__ neighbors,
    rbmd::Id random_neighbor_capacity,
    rbmd::Id* __restrict__ random_neighbors,
    rbmd::Id* __restrict__ random_neighbors_num,
    rbmd::Real* __restrict__ random_neighbor_weight,
    rbmd::Id* __restrict__ required_random_neighbor_capacity, Box box,
    rbmd::Id* __restrict__ should_realloc,
    rbmd::Id* __restrict__ required_neighbor_capacity,
    rbmd::Id selection_frequency, rbmd::Id sample_epoch) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  __shared__ typename WarpScan::TempStorage
      core_scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];
  __shared__ typename WarpScan::TempStorage
      shell_scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];

  const rbmd::Id group_in_block =
      threadIdx.x / static_cast<rbmd::Id>(THREADS_PER_PARTICLE);
  const rbmd::Id lane =
      threadIdx.x % static_cast<rbmd::Id>(THREADS_PER_PARTICLE);
  const rbmd::Id atom_idx =
      blockIdx.x * (blockDim.x / THREADS_PER_PARTICLE) + group_in_block;
  const bool active = atom_idx < total_atom_num;

  const rbmd::Id atom_id = active ? __ldg(&atoms_id[atom_idx]) : rbmd::Id(0);
  const rbmd::Id neighbor_row_start =
      active ? __ldg(&neighbor_start[atom_idx]) : rbmd::Id(0);
  const rbmd::Id cell_idx =
      active ? __ldg(&per_atom_cell_id[atom_idx]) : rbmd::Id(0);
  const std::uint64_t neighbor_cell_row =
      active ? static_cast<std::uint64_t>(cell_idx) *
                   static_cast<std::uint64_t>(neighbor_cell_num)
             : std::uint64_t(0);
  const rbmd::Real center_x =
      active ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_y =
      active ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_z =
      active ? __ldg(&pz[atom_idx]) : rbmd::Real(0);

  rbmd::Id total_neighbor_num = 0;
  rbmd::Id shell_neighbor_num = 0;
  for (rbmd::Id cell_offset = 0; cell_offset < neighbor_cell_num;
       ++cell_offset) {
    const rbmd::Id neighbor_cell_idx =
        active ? __ldg(&neighbor_cell[
                     neighbor_cell_row +
                     static_cast<std::uint64_t>(cell_offset)])
               : rbmd::Id(0);
    const rbmd::Id start =
        active ? __ldg(&in_atom_list_start_index[neighbor_cell_idx])
               : rbmd::Id(0);
    const rbmd::Id end =
        active ? __ldg(&in_atom_list_end_index[neighbor_cell_idx])
               : rbmd::Id(0);
    const rbmd::Id cell_size = end - start;
    for (rbmd::Id base = 0; base < cell_size;
         base += THREADS_PER_PARTICLE) {
      const rbmd::Id neighbor_atom_pos = start + base + lane;
      const bool valid = active && neighbor_atom_pos < end;
      const rbmd::Id neighbor_atom_idx =
          valid ? __ldg(&sorted_atom_indices[neighbor_atom_pos]) : rbmd::Id(0);
      bool core_accepted = false;
      bool shell_accepted = false;
      bool shell_selected = false;
      if (valid && atom_idx != neighbor_atom_idx) {
        const rbmd::Real distance = CaculateDistance(
            box, center_x, center_y, center_z, __ldg(&px[neighbor_atom_idx]),
            __ldg(&py[neighbor_atom_idx]), __ldg(&pz[neighbor_atom_idx]));
        core_accepted = distance < trunc_distance_power_2;
        shell_accepted = !core_accepted && distance < cutoff_2;
        shell_selected =
            shell_accepted &&
            RblShellSampleSelected(
                selection_frequency, sample_epoch, atom_id,
                __ldg(&atoms_id[neighbor_atom_idx]));
      }

      rbmd::Id core_rank = 0;
      rbmd::Id core_count = 0;
      WarpScan(core_scan_storage[group_in_block])
          .ExclusiveSum(static_cast<rbmd::Id>(core_accepted), core_rank,
                        core_count);
      if (core_accepted &&
          total_neighbor_num + core_rank < max_neighbor_num) {
        neighbors[neighbor_row_start + total_neighbor_num + core_rank] =
            neighbor_atom_idx;
      }
      total_neighbor_num += core_count;

      rbmd::Id shell_rank = 0;
      rbmd::Id shell_count = 0;
      WarpScan(shell_scan_storage[group_in_block])
          .ExclusiveSum(static_cast<rbmd::Id>(shell_selected), shell_rank,
                        shell_count);
      const rbmd::Id sample_slot = shell_neighbor_num + shell_rank;

      if (shell_count > 0) {
#if defined(__CUDA)
        // Only the lanes of this particle group enter the shuffle loop.
        const unsigned int group_mask =
            (0xffffffffu >> (32 - THREADS_PER_PARTICLE))
            << ((threadIdx.x % 32 / THREADS_PER_PARTICLE) *
                THREADS_PER_PARTICLE);
#endif
#pragma unroll
        for (int source_lane = 0; source_lane < THREADS_PER_PARTICLE;
             ++source_lane) {
#if defined(__CUDA)
          const rbmd::Id ordered_slot = __shfl_sync(
              group_mask, sample_slot, source_lane, THREADS_PER_PARTICLE);
          const rbmd::Id ordered_neighbor = __shfl_sync(
              group_mask, neighbor_atom_idx, source_lane,
              THREADS_PER_PARTICLE);
          const int ordered_accept = __shfl_sync(
              group_mask, static_cast<int>(shell_selected), source_lane,
              THREADS_PER_PARTICLE);
#else
          const rbmd::Id ordered_slot =
              __shfl(sample_slot, source_lane, THREADS_PER_PARTICLE);
          const rbmd::Id ordered_neighbor =
              __shfl(neighbor_atom_idx, source_lane, THREADS_PER_PARTICLE);
          const int ordered_accept = __shfl(
              static_cast<int>(shell_selected), source_lane,
              THREADS_PER_PARTICLE);
#endif
          if (lane == 0 && ordered_accept != 0 &&
              ordered_slot < random_neighbor_capacity) {
            const unsigned long long output =
                static_cast<unsigned long long>(ordered_slot) *
                    static_cast<unsigned long long>(total_atom_num) +
                static_cast<unsigned long long>(atom_idx);
            random_neighbors[output] = ordered_neighbor;
          }
        }
      }
      shell_neighbor_num += shell_count;
    }
    // Different particle groups can traverse cells with different sizes.
    // Re-converge the physical wave before the next cell so subgroup scratch
    // storage cannot be reused while another group is still in this cell.
    SYNCWARP(0xFFFFFFFF);
  }

  if (active && lane == 0) {
    const rbmd::Id written_neighbor_num =
        MIN(total_neighbor_num, max_neighbor_num);
    neighbor_end[atom_idx] = neighbor_row_start + written_neighbor_num;
    actual_neighbor_num[atom_idx] = total_neighbor_num;
    const rbmd::Id sampled_neighbor_num =
        MIN(shell_neighbor_num, random_neighbor_capacity);
    random_neighbors_num[atom_idx] = sampled_neighbor_num;
    random_neighbor_weight[atom_idx] = static_cast<rbmd::Real>(
        selection_frequency > 0 ? selection_frequency : rbmd::Id(1));
    rbmd::AtomicMaxCount(required_random_neighbor_capacity, shell_neighbor_num);
    rbmd::AtomicMaxCount(required_neighbor_capacity, total_neighbor_num);
    if (total_neighbor_num > max_neighbor_num) {
      rbmd::AtomicOrCount(should_realloc, RBMD_TRUE);
    }
  }
}

__device__ __forceinline__ bool RblCompactIndexInRange(rbmd::Id value,
                                                       rbmd::Id upper) {
  return static_cast<unsigned long long>(value) <
         static_cast<unsigned long long>(upper);
}

__device__ __forceinline__ void SetRblCompactError(
    rbmd::Id* error_code, rbmd::Id code) {
  rbmd::AtomicMaxCount(error_code, code);
}

template <int THREADS_PER_PARTICLE>
__global__ void CountRblCompactCandidatesGrouped(
    const rbmd::Id* __restrict__ per_atom_cell_id,
    const rbmd::Id* __restrict__ in_atom_list_start_index,
    const rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real cutoff_2, rbmd::Id active_atom_num,
    rbmd::Id total_storage_atoms, rbmd::Id total_cells,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Id* __restrict__ sorted_atom_indices,
    const rbmd::Real* __restrict__ sorted_px,
    const rbmd::Real* __restrict__ sorted_py,
    const rbmd::Real* __restrict__ sorted_pz,
    const rbmd::Id* __restrict__ neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id* __restrict__ candidate_num,
    rbmd::Id* __restrict__ max_candidate_num, Box box,
    rbmd::Id* __restrict__ error_code) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  __shared__ typename WarpScan::TempStorage
      scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];
  const rbmd::Id group_in_block = threadIdx.x / THREADS_PER_PARTICLE;
  const rbmd::Id lane = threadIdx.x % THREADS_PER_PARTICLE;
  const rbmd::Id atom_idx =
      blockIdx.x * (blockDim.x / THREADS_PER_PARTICLE) + group_in_block;
  const bool active = RblCompactIndexInRange(atom_idx, active_atom_num);
  rbmd::Id cell_idx = 0;
  bool traversal_valid = active;
  if (active) {
    cell_idx = __ldg(&per_atom_cell_id[atom_idx]);
    if (!RblCompactIndexInRange(cell_idx, total_cells)) {
      SetRblCompactError(error_code, rbmd::Id(1));
      traversal_valid = false;
    }
  }
  const rbmd::Real center_x =
      traversal_valid ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_y =
      traversal_valid ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_z =
      traversal_valid ? __ldg(&pz[atom_idx]) : rbmd::Real(0);
  const std::uint64_t stencil_row =
      static_cast<std::uint64_t>(cell_idx) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  const std::uint64_t stencil_storage =
      static_cast<std::uint64_t>(total_cells) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  rbmd::Id count = 0;
  for (rbmd::Id cell_offset = 0; cell_offset < neighbor_cell_num;
       ++cell_offset) {
    const std::uint64_t stencil_pos =
        stencil_row + static_cast<std::uint64_t>(cell_offset);
    rbmd::Id start = 0;
    rbmd::Id end = 0;
    if (traversal_valid && stencil_pos >= stencil_storage) {
      SetRblCompactError(error_code, rbmd::Id(2));
      traversal_valid = false;
    }
    if (traversal_valid) {
      const rbmd::Id neighbor_cell_idx = __ldg(&neighbor_cell[stencil_pos]);
      if (!RblCompactIndexInRange(neighbor_cell_idx, total_cells)) {
        SetRblCompactError(error_code, rbmd::Id(2));
        traversal_valid = false;
      } else {
        start = __ldg(&in_atom_list_start_index[neighbor_cell_idx]);
        end = __ldg(&in_atom_list_end_index[neighbor_cell_idx]);
        if (static_cast<std::uint64_t>(start) >
                static_cast<std::uint64_t>(end) ||
            static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(total_storage_atoms)) {
          SetRblCompactError(error_code, rbmd::Id(3));
          traversal_valid = false;
          start = 0;
          end = 0;
        }
      }
    }
    const rbmd::Id cell_size = end - start;
    for (rbmd::Id base = 0; base < cell_size;
         base += THREADS_PER_PARTICLE) {
      const rbmd::Id pos = start + base + lane;
      rbmd::Id accepted = 0;
      if (pos < end) {
        const rbmd::Id neighbor_atom_idx = __ldg(&sorted_atom_indices[pos]);
        if (!RblCompactIndexInRange(neighbor_atom_idx,
                                    total_storage_atoms)) {
          SetRblCompactError(error_code, rbmd::Id(4));
        } else if (neighbor_atom_idx != atom_idx &&
                   CaculateDistance(box, center_x, center_y, center_z,
                                    __ldg(&sorted_px[pos]),
                                    __ldg(&sorted_py[pos]),
                                    __ldg(&sorted_pz[pos])) < cutoff_2) {
          accepted = 1;
        }
      }
      rbmd::Id compact_offset = 0;
      rbmd::Id accepted_in_batch = 0;
      WarpScan(scan_storage[group_in_block])
          .ExclusiveSum(accepted, compact_offset, accepted_in_batch);
      count += accepted_in_batch;
    }
  }
  if (active && lane == 0) {
    candidate_num[atom_idx] = traversal_valid ? count : rbmd::Id(0);
    if (traversal_valid) {
      rbmd::AtomicMaxCount(max_candidate_num, count);
    }
  }
}

__device__ __forceinline__ rbmd::Id RblAtomicAddId(rbmd::Id* address,
                                                    rbmd::Id value) {
#if USE_64BIT_IDS
  return static_cast<rbmd::Id>(atomicAdd(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(value)));
#else
  return static_cast<rbmd::Id>(atomicAdd(
      reinterpret_cast<unsigned int*>(address),
      static_cast<unsigned int>(value)));
#endif
}

template <int THREADS_PER_PARTICLE, int MAX_CANDIDATES_PER_ATOM>
__global__ void AppendRblCompactCandidatesGrouped(
    const rbmd::Id* __restrict__ per_atom_cell_id,
    const rbmd::Id* __restrict__ in_atom_list_start_index,
    const rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real cutoff_2, rbmd::Id active_atom_num,
    rbmd::Id total_storage_atoms, rbmd::Id total_cells,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Id* __restrict__ sorted_atom_indices,
    const rbmd::Real* __restrict__ sorted_px,
    const rbmd::Real* __restrict__ sorted_py,
    const rbmd::Real* __restrict__ sorted_pz,
    const rbmd::Id* __restrict__ neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id* __restrict__ candidate_num,
    rbmd::Id* __restrict__ candidate_start,
    rbmd::Id* __restrict__ candidate_end,
    rbmd::Id* __restrict__ candidates,
    std::uint64_t candidate_storage_size,
    rbmd::Id* __restrict__ total_candidate_num,
    rbmd::Id* __restrict__ max_candidate_num, Box box,
    rbmd::Id* __restrict__ error_code) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  constexpr int groups_per_block = BLOCK_SIZE / THREADS_PER_PARTICLE;
  __shared__ typename WarpScan::TempStorage scan_storage[groups_per_block];
  __shared__ rbmd::Id
      buffered_candidates[groups_per_block][MAX_CANDIDATES_PER_ATOM];
  __shared__ rbmd::Id group_counts[groups_per_block];
  __shared__ rbmd::Id group_offsets[groups_per_block];
  __shared__ rbmd::Id block_base;
  __shared__ rbmd::Id block_write_valid;

  const rbmd::Id group = threadIdx.x / THREADS_PER_PARTICLE;
  const rbmd::Id lane = threadIdx.x % THREADS_PER_PARTICLE;
  const rbmd::Id atom_idx =
      blockIdx.x * groups_per_block + group;
  const bool active = RblCompactIndexInRange(atom_idx, active_atom_num);
  rbmd::Id cell_idx = 0;
  bool traversal_valid = active;
  if (active) {
    cell_idx = __ldg(&per_atom_cell_id[atom_idx]);
    if (!RblCompactIndexInRange(cell_idx, total_cells)) {
      SetRblCompactError(error_code, rbmd::Id(40));
      traversal_valid = false;
    }
  }
  const rbmd::Real center_x =
      traversal_valid ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_y =
      traversal_valid ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_z =
      traversal_valid ? __ldg(&pz[atom_idx]) : rbmd::Real(0);
  const std::uint64_t stencil_row =
      static_cast<std::uint64_t>(cell_idx) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  const std::uint64_t stencil_storage =
      static_cast<std::uint64_t>(total_cells) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  rbmd::Id count = 0;
  for (rbmd::Id cell_offset = 0; cell_offset < neighbor_cell_num;
       ++cell_offset) {
    const std::uint64_t stencil_pos =
        stencil_row + static_cast<std::uint64_t>(cell_offset);
    rbmd::Id start = 0;
    rbmd::Id end = 0;
    if (traversal_valid && stencil_pos >= stencil_storage) {
      SetRblCompactError(error_code, rbmd::Id(41));
      traversal_valid = false;
    }
    if (traversal_valid) {
      const rbmd::Id neighbor_cell_idx = __ldg(&neighbor_cell[stencil_pos]);
      if (!RblCompactIndexInRange(neighbor_cell_idx, total_cells)) {
        SetRblCompactError(error_code, rbmd::Id(41));
        traversal_valid = false;
      } else {
        start = __ldg(&in_atom_list_start_index[neighbor_cell_idx]);
        end = __ldg(&in_atom_list_end_index[neighbor_cell_idx]);
        if (static_cast<std::uint64_t>(start) >
                static_cast<std::uint64_t>(end) ||
            static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(total_storage_atoms)) {
          SetRblCompactError(error_code, rbmd::Id(42));
          traversal_valid = false;
          start = 0;
          end = 0;
        }
      }
    }
    const rbmd::Id cell_size = end - start;
    for (rbmd::Id base = 0; base < cell_size;
         base += THREADS_PER_PARTICLE) {
      const rbmd::Id pos = start + base + lane;
      rbmd::Id neighbor_atom_idx = 0;
      rbmd::Id accepted = 0;
      if (pos < end) {
        neighbor_atom_idx = __ldg(&sorted_atom_indices[pos]);
        if (!RblCompactIndexInRange(neighbor_atom_idx,
                                    total_storage_atoms)) {
          SetRblCompactError(error_code, rbmd::Id(43));
        } else if (neighbor_atom_idx != atom_idx &&
                   CaculateDistance(box, center_x, center_y, center_z,
                                    __ldg(&sorted_px[pos]),
                                    __ldg(&sorted_py[pos]),
                                    __ldg(&sorted_pz[pos])) < cutoff_2) {
          accepted = 1;
        }
      }
      rbmd::Id compact_offset = 0;
      rbmd::Id accepted_in_batch = 0;
      WarpScan(scan_storage[group])
          .ExclusiveSum(accepted, compact_offset, accepted_in_batch);
      if (accepted && count + compact_offset < MAX_CANDIDATES_PER_ATOM) {
        buffered_candidates[group][count + compact_offset] = neighbor_atom_idx;
      }
      count += accepted_in_batch;
    }
  }

  if (lane == 0) {
    rbmd::Id stored_count = 0;
    if (active && traversal_valid) {
      if (count <= MAX_CANDIDATES_PER_ATOM) {
        stored_count = count;
      } else {
        SetRblCompactError(error_code, rbmd::Id(44));
      }
    }
    group_counts[group] = stored_count;
    // Preserve the measured count separately from stored_count: an overflowing
    // row is intentionally excluded from the block write, but its true size is
    // still needed by the host-side fallback. group_offsets is overwritten with
    // prefix offsets by thread 0 after the block barrier below.
    group_offsets[group] =
        active && traversal_valid ? count : rbmd::Id(0);
    if (active) {
      candidate_num[atom_idx] = stored_count;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    rbmd::Id block_total = 0;
    rbmd::Id block_max = 0;
    for (int g = 0; g < groups_per_block; ++g) {
      block_max = MAX(block_max, group_offsets[g]);
      group_offsets[g] = block_total;
      block_total += group_counts[g];
    }
    if (block_max > 0) {
      rbmd::AtomicMaxCount(max_candidate_num, block_max);
    }
    const rbmd::Id base = block_total > 0
                              ? RblAtomicAddId(total_candidate_num, block_total)
                              : rbmd::Id(0);
    block_base = base;
    block_write_valid =
        base >= 0 &&
                static_cast<std::uint64_t>(base) <= candidate_storage_size &&
                static_cast<std::uint64_t>(block_total) <=
                    candidate_storage_size - static_cast<std::uint64_t>(base)
            ? RBMD_TRUE
            : RBMD_FALSE;
    if (block_write_valid == RBMD_FALSE) {
      SetRblCompactError(error_code, rbmd::Id(45));
    }
  }
  __syncthreads();

  if (active) {
    const rbmd::Id row_count = group_counts[group];
    const rbmd::Id row_start = block_base + group_offsets[group];
    if (lane == 0) {
      candidate_start[atom_idx] = row_count > 0 ? row_start : rbmd::Id(0);
      candidate_end[atom_idx] =
          row_count > 0 ? row_start + row_count : rbmd::Id(0);
    }
    if (block_write_valid != RBMD_FALSE) {
      for (rbmd::Id i = lane; i < row_count;
           i += THREADS_PER_PARTICLE) {
        candidates[static_cast<std::uint64_t>(row_start) + i] =
            buffered_candidates[group][i];
      }
    }
  }
}

template <int THREADS_PER_PARTICLE>
__global__ void FillRblCompactCandidatesGrouped(
    const rbmd::Id* __restrict__ per_atom_cell_id,
    const rbmd::Id* __restrict__ in_atom_list_start_index,
    const rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real cutoff_2, rbmd::Id active_atom_num,
    rbmd::Id total_storage_atoms, rbmd::Id total_cells,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Id* __restrict__ sorted_atom_indices,
    const rbmd::Real* __restrict__ sorted_px,
    const rbmd::Real* __restrict__ sorted_py,
    const rbmd::Real* __restrict__ sorted_pz,
    const rbmd::Id* __restrict__ neighbor_cell,
    rbmd::Id neighbor_cell_num, const rbmd::Id* __restrict__ candidate_num,
    const rbmd::Id* __restrict__ candidate_start,
    rbmd::Id* __restrict__ candidate_end, rbmd::Id* __restrict__ candidates,
    std::uint64_t candidate_storage_size, Box box,
    rbmd::Id* __restrict__ error_code) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  __shared__ typename WarpScan::TempStorage
      scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];
  const rbmd::Id group_in_block = threadIdx.x / THREADS_PER_PARTICLE;
  const rbmd::Id lane = threadIdx.x % THREADS_PER_PARTICLE;
  const rbmd::Id atom_idx =
      blockIdx.x * (blockDim.x / THREADS_PER_PARTICLE) + group_in_block;
  const bool active = RblCompactIndexInRange(atom_idx, active_atom_num);
  const rbmd::Id expected = active ? __ldg(&candidate_num[atom_idx]) : 0;
  const rbmd::Id output_start =
      active ? __ldg(&candidate_start[atom_idx]) : 0;
  const std::uint64_t output_begin = static_cast<std::uint64_t>(output_start);
  bool traversal_valid = active;
  if (active &&
      (output_begin > candidate_storage_size ||
       static_cast<std::uint64_t>(expected) >
           candidate_storage_size - output_begin)) {
    SetRblCompactError(error_code, rbmd::Id(14));
    traversal_valid = false;
  }
  rbmd::Id cell_idx = 0;
  if (traversal_valid) {
    cell_idx = __ldg(&per_atom_cell_id[atom_idx]);
    if (!RblCompactIndexInRange(cell_idx, total_cells)) {
      SetRblCompactError(error_code, rbmd::Id(10));
      traversal_valid = false;
    }
  }
  const rbmd::Real center_x =
      traversal_valid ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_y =
      traversal_valid ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_z =
      traversal_valid ? __ldg(&pz[atom_idx]) : rbmd::Real(0);
  const std::uint64_t stencil_row =
      static_cast<std::uint64_t>(cell_idx) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  const std::uint64_t stencil_storage =
      static_cast<std::uint64_t>(total_cells) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  rbmd::Id written = 0;
  for (rbmd::Id cell_offset = 0; cell_offset < neighbor_cell_num;
       ++cell_offset) {
    const std::uint64_t stencil_pos =
        stencil_row + static_cast<std::uint64_t>(cell_offset);
    rbmd::Id start = 0;
    rbmd::Id end = 0;
    if (traversal_valid && stencil_pos >= stencil_storage) {
      SetRblCompactError(error_code, rbmd::Id(11));
      traversal_valid = false;
    }
    if (traversal_valid) {
      const rbmd::Id neighbor_cell_idx = __ldg(&neighbor_cell[stencil_pos]);
      if (!RblCompactIndexInRange(neighbor_cell_idx, total_cells)) {
        SetRblCompactError(error_code, rbmd::Id(11));
        traversal_valid = false;
      } else {
        start = __ldg(&in_atom_list_start_index[neighbor_cell_idx]);
        end = __ldg(&in_atom_list_end_index[neighbor_cell_idx]);
        if (static_cast<std::uint64_t>(start) >
                static_cast<std::uint64_t>(end) ||
            static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(total_storage_atoms)) {
          SetRblCompactError(error_code, rbmd::Id(12));
          traversal_valid = false;
          start = 0;
          end = 0;
        }
      }
    }
    const rbmd::Id cell_size = end - start;
    for (rbmd::Id base = 0; base < cell_size;
         base += THREADS_PER_PARTICLE) {
      const rbmd::Id pos = start + base + lane;
      rbmd::Id neighbor_atom_idx = 0;
      rbmd::Id accepted = 0;
      if (pos < end) {
        neighbor_atom_idx = __ldg(&sorted_atom_indices[pos]);
        if (!RblCompactIndexInRange(neighbor_atom_idx,
                                    total_storage_atoms)) {
          SetRblCompactError(error_code, rbmd::Id(13));
        } else if (neighbor_atom_idx != atom_idx &&
                   CaculateDistance(box, center_x, center_y, center_z,
                                    __ldg(&sorted_px[pos]),
                                    __ldg(&sorted_py[pos]),
                                    __ldg(&sorted_pz[pos])) < cutoff_2) {
          accepted = 1;
        }
      }
      rbmd::Id compact_offset = 0;
      rbmd::Id accepted_in_batch = 0;
      WarpScan(scan_storage[group_in_block])
          .ExclusiveSum(accepted, compact_offset, accepted_in_batch);
      if (accepted) {
        const std::uint64_t output = output_begin +
                                     static_cast<std::uint64_t>(written) +
                                     static_cast<std::uint64_t>(compact_offset);
        if (written + compact_offset >= expected ||
            output >= candidate_storage_size) {
          SetRblCompactError(error_code, rbmd::Id(14));
        } else {
          candidates[output] = neighbor_atom_idx;
        }
      }
      written += accepted_in_batch;
    }
  }
  if (active && lane == 0) {
    if (!traversal_valid || written != expected) {
      SetRblCompactError(error_code, rbmd::Id(15));
    } else {
      candidate_end[atom_idx] = output_start + written;
    }
  }
}

template <int THREADS_PER_PARTICLE>
__global__ void FillRblFixedCandidatesGrouped(
    const rbmd::Id* __restrict__ per_atom_cell_id,
    const rbmd::Id* __restrict__ in_atom_list_start_index,
    const rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real cutoff_2, rbmd::Real core_cutoff_2,
    rbmd::Id active_atom_num,
    rbmd::Id total_storage_atoms, rbmd::Id total_cells,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Id* __restrict__ sorted_atom_indices,
    const rbmd::Real* __restrict__ sorted_px,
    const rbmd::Real* __restrict__ sorted_py,
    const rbmd::Real* __restrict__ sorted_pz,
    const rbmd::Id* __restrict__ neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id candidate_capacity,
    const rbmd::Id* __restrict__ candidate_start,
    rbmd::Id* __restrict__ candidate_end, rbmd::Id* __restrict__ candidates,
    rbmd::Id* __restrict__ candidate_num,
    bool implicit_offsets, std::uint64_t candidate_storage_size,
    rbmd::Id core_capacity, const rbmd::Id* __restrict__ core_start,
    rbmd::Id* __restrict__ core_end, rbmd::Id* __restrict__ core_neighbors,
    rbmd::Id* __restrict__ core_neighbor_num,
    std::uint64_t core_storage_size, Box box,
    rbmd::Id* __restrict__ status) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  __shared__ typename WarpScan::TempStorage
      scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];
  __shared__ typename WarpScan::TempStorage
      core_scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];
  __shared__ rbmd::Id
      block_candidate_count[BLOCK_SIZE / THREADS_PER_PARTICLE];
  __shared__ rbmd::Id
      block_core_count[BLOCK_SIZE / THREADS_PER_PARTICLE];
  __shared__ rbmd::Id
      block_group_valid[BLOCK_SIZE / THREADS_PER_PARTICLE];
  const rbmd::Id group = threadIdx.x / THREADS_PER_PARTICLE;
  const rbmd::Id lane = threadIdx.x % THREADS_PER_PARTICLE;
  const rbmd::Id atom_idx =
      blockIdx.x * (blockDim.x / THREADS_PER_PARTICLE) + group;
  const bool active = RblCompactIndexInRange(atom_idx, active_atom_num);
  const std::uint64_t output_start =
      active ? (implicit_offsets
                    ? static_cast<std::uint64_t>(atom_idx) *
                          static_cast<std::uint64_t>(candidate_capacity)
                    : static_cast<std::uint64_t>(
                          __ldg(&candidate_start[atom_idx])))
             : std::uint64_t(0);
  const bool core_mode = core_cutoff_2 >= rbmd::Real(0) &&
      core_start != nullptr && core_end != nullptr &&
      core_neighbors != nullptr && core_neighbor_num != nullptr;
  const bool generate_core = active && core_mode;
  const std::uint64_t core_output_start =
      generate_core ? static_cast<std::uint64_t>(__ldg(&core_start[atom_idx]))
                    : std::uint64_t(0);
  bool valid = active;
  const bool candidate_row_out_of_bounds =
      output_start > candidate_storage_size ||
      static_cast<std::uint64_t>(candidate_capacity) >
          candidate_storage_size - output_start;
  const bool core_row_out_of_bounds =
      generate_core &&
      (core_output_start > core_storage_size ||
       static_cast<std::uint64_t>(core_capacity) >
           core_storage_size - core_output_start);
  if (valid && (candidate_row_out_of_bounds || core_row_out_of_bounds)) {
    SetRblCompactError(&status[4], rbmd::Id(25));
    valid = false;
  }
  rbmd::Id cell_idx = 0;
  if (valid) {
    cell_idx = __ldg(&per_atom_cell_id[atom_idx]);
    if (!RblCompactIndexInRange(cell_idx, total_cells)) {
      SetRblCompactError(&status[4], rbmd::Id(20));
      valid = false;
    }
  }
  const rbmd::Real x = valid ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real y = valid ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real z = valid ? __ldg(&pz[atom_idx]) : rbmd::Real(0);
  const std::uint64_t stencil_row =
      static_cast<std::uint64_t>(cell_idx) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  const std::uint64_t stencil_size =
      static_cast<std::uint64_t>(total_cells) *
      static_cast<std::uint64_t>(neighbor_cell_num);
  rbmd::Id written = 0;
  rbmd::Id core_written = 0;
  for (rbmd::Id cell_offset = 0; cell_offset < neighbor_cell_num;
       ++cell_offset) {
    const std::uint64_t stencil_pos =
        stencil_row + static_cast<std::uint64_t>(cell_offset);
    rbmd::Id start = 0;
    rbmd::Id end = 0;
    if (valid && stencil_pos >= stencil_size) {
      SetRblCompactError(&status[4], rbmd::Id(21));
      valid = false;
    }
    if (valid) {
      const rbmd::Id neighbor_cell_idx = __ldg(&neighbor_cell[stencil_pos]);
      if (!RblCompactIndexInRange(neighbor_cell_idx, total_cells)) {
        SetRblCompactError(&status[4], rbmd::Id(21));
        valid = false;
      } else {
        start = __ldg(&in_atom_list_start_index[neighbor_cell_idx]);
        end = __ldg(&in_atom_list_end_index[neighbor_cell_idx]);
        if (start > end || !RblCompactIndexInRange(end, total_storage_atoms + 1)) {
          SetRblCompactError(&status[4], rbmd::Id(22));
          valid = false;
          start = end = 0;
        }
      }
    }
    for (rbmd::Id base = 0; base < end - start;
         base += THREADS_PER_PARTICLE) {
      const rbmd::Id pos = start + base + lane;
      rbmd::Id neighbor_idx = 0;
      rbmd::Id accepted = 0;
      rbmd::Id core_accepted = 0;
      if (pos < end) {
        neighbor_idx = __ldg(&sorted_atom_indices[pos]);
        if (!RblCompactIndexInRange(neighbor_idx, total_storage_atoms)) {
          SetRblCompactError(&status[4], rbmd::Id(23));
        } else if (neighbor_idx != atom_idx) {
          const rbmd::Real distance = CaculateDistance(
              box, x, y, z, __ldg(&sorted_px[pos]),
              __ldg(&sorted_py[pos]), __ldg(&sorted_pz[pos]));
          accepted = distance < cutoff_2 ? rbmd::Id(1) : rbmd::Id(0);
          core_accepted =
              generate_core && accepted && distance < core_cutoff_2
                  ? rbmd::Id(1)
                  : rbmd::Id(0);
        }
      }
      rbmd::Id offset = 0;
      rbmd::Id batch_count = 0;
      WarpScan(scan_storage[group]).ExclusiveSum(accepted, offset, batch_count);
      if (accepted && written + offset < candidate_capacity) {
        const std::uint64_t output =
            output_start + static_cast<std::uint64_t>(written + offset);
        if (output < candidate_storage_size) {
          candidates[output] = neighbor_idx;
        } else {
          SetRblCompactError(&status[4], rbmd::Id(24));
        }
      }
      written += batch_count;

      if (core_mode) {
        rbmd::Id core_offset = 0;
        rbmd::Id core_batch_count = 0;
        WarpScan(core_scan_storage[group])
            .ExclusiveSum(core_accepted, core_offset, core_batch_count);
        if (core_accepted && core_written + core_offset < core_capacity) {
          const std::uint64_t output =
              core_output_start +
              static_cast<std::uint64_t>(core_written + core_offset);
          if (output < core_storage_size) {
            core_neighbors[output] = neighbor_idx;
          } else {
            SetRblCompactError(&status[4], rbmd::Id(26));
          }
        }
        core_written += core_batch_count;
      }
    }
  }
  if (active && lane == 0) {
    candidate_num[atom_idx] = written;
    if (!implicit_offsets) {
      candidate_end[atom_idx] = static_cast<rbmd::Id>(output_start) +
                                MIN(written, candidate_capacity);
    }
    if (generate_core) {
      core_neighbor_num[atom_idx] = core_written;
      core_end[atom_idx] = static_cast<rbmd::Id>(core_output_start) +
                           MIN(core_written, core_capacity);
    }
  }
  if (lane == 0) {
    block_candidate_count[group] = active ? written : rbmd::Id(0);
    block_core_count[group] = generate_core ? core_written : rbmd::Id(0);
    block_group_valid[group] = !active || valid ? RBMD_TRUE : RBMD_FALSE;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    rbmd::Id block_candidate_max = 0;
    rbmd::Id block_core_max = 0;
    rbmd::Id candidate_realloc = RBMD_FALSE;
    rbmd::Id core_realloc = RBMD_FALSE;
    for (rbmd::Id i = 0; i < BLOCK_SIZE / THREADS_PER_PARTICLE; ++i) {
      block_candidate_max =
          MAX(block_candidate_max, block_candidate_count[i]);
      block_core_max = MAX(block_core_max, block_core_count[i]);
      if (block_candidate_count[i] > candidate_capacity) {
        candidate_realloc = RBMD_TRUE;
      }
      if (core_mode && block_core_count[i] > core_capacity) {
        core_realloc = RBMD_TRUE;
      }
      if (block_group_valid[i] == RBMD_FALSE) {
        candidate_realloc = RBMD_TRUE;
        core_realloc = RBMD_TRUE;
      }
    }
    rbmd::AtomicMaxCount(&status[0], block_candidate_max);
    if (core_mode) {
      rbmd::AtomicMaxCount(&status[1], block_core_max);
    }
    if (candidate_realloc == RBMD_TRUE) {
      rbmd::AtomicOrCount(&status[2], RBMD_TRUE);
    }
    if (core_realloc == RBMD_TRUE) {
      rbmd::AtomicOrCount(&status[3], RBMD_TRUE);
    }
  }
}

__global__ void FilterRBLNeighborCandidates(
    rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* __restrict__ atoms_id,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Id* __restrict__ candidate_start,
    const rbmd::Id* __restrict__ candidate_end,
    const rbmd::Id* __restrict__ candidate_neighbors,
    const rbmd::Id* __restrict__ candidate_num,
    rbmd::Id candidate_capacity, bool implicit_offsets,
    const rbmd::Id* __restrict__ max_core_neighbor_num,
    const rbmd::Id* __restrict__ core_start,
    rbmd::Id* __restrict__ core_end, rbmd::Id* __restrict__ core_neighbors,
    rbmd::Id* __restrict__ core_neighbor_num,
    rbmd::Id random_neighbor_capacity,
    rbmd::Id* __restrict__ random_neighbors,
    rbmd::Id* __restrict__ random_neighbors_num,
    rbmd::Id* __restrict__ required_random_neighbor_capacity,
    rbmd::Id* __restrict__ required_core_neighbor_capacity, Box box,
    rbmd::Id* __restrict__ should_realloc, rbmd::Id selection_frequency,
    rbmd::Id sample_epoch) {
  const rbmd::Id atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom_idx >= total_atom_num) {
    return;
  }

  const rbmd::Real x = __ldg(&px[atom_idx]);
  const rbmd::Real y = __ldg(&py[atom_idx]);
  const rbmd::Real z = __ldg(&pz[atom_idx]);
  rbmd::Id core_count = 0;
  rbmd::Id sampled_count = 0;
  const rbmd::Id output_start = core_start[atom_idx];
  const rbmd::Id core_capacity = max_core_neighbor_num[atom_idx];

  const std::uint64_t candidate_begin =
      implicit_offsets
          ? static_cast<std::uint64_t>(atom_idx) *
                static_cast<std::uint64_t>(candidate_capacity)
          : static_cast<std::uint64_t>(candidate_start[atom_idx]);
  const std::uint64_t candidate_finish =
      implicit_offsets
          ? candidate_begin + static_cast<std::uint64_t>(candidate_num[atom_idx])
          : static_cast<std::uint64_t>(candidate_end[atom_idx]);
  for (std::uint64_t pos = candidate_begin; pos < candidate_finish; ++pos) {
    const rbmd::Id neighbor_idx = __ldg(&candidate_neighbors[pos]);
    if (neighbor_idx == atom_idx) {
      continue;
    }
    const rbmd::Real distance = CaculateDistance(
        box, x, y, z, __ldg(&px[neighbor_idx]), __ldg(&py[neighbor_idx]),
        __ldg(&pz[neighbor_idx]));
    if (distance < core_cutoff_2) {
      if (core_count < core_capacity) {
        core_neighbors[output_start + core_count] = neighbor_idx;
      } else {
        rbmd::AtomicOrCount(should_realloc, RBMD_TRUE);
      }
      ++core_count;
    } else if (distance < force_cutoff_2 &&
               RblShellSampleSelected(selection_frequency, sample_epoch,
                                      __ldg(&atoms_id[atom_idx]),
                                      __ldg(&atoms_id[neighbor_idx]))) {
      if (sampled_count < random_neighbor_capacity) {
        const unsigned long random_neighbor_index =
            static_cast<unsigned long>(sampled_count) *
                static_cast<unsigned long>(total_atom_num) +
            static_cast<unsigned long>(atom_idx);
        random_neighbors[random_neighbor_index] = neighbor_idx;
      }
      ++sampled_count;
    }
  }

  core_end[atom_idx] = output_start +
                       ((core_count < core_capacity) ? core_count
                                                     : core_capacity);
  core_neighbor_num[atom_idx] = core_count;
  random_neighbors_num[atom_idx] =
      MIN(sampled_count, random_neighbor_capacity);
  rbmd::AtomicMaxCount(required_random_neighbor_capacity, sampled_count);
  rbmd::AtomicMaxCount(required_core_neighbor_capacity, core_count);
}

template <int THREADS_PER_PARTICLE>
__global__ void FilterRBLCoreCandidatesGrouped(
    rbmd::Real core_cutoff_2, rbmd::Id total_atom_num,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Id* __restrict__ candidate_start,
    const rbmd::Id* __restrict__ candidate_end,
    const rbmd::Id* __restrict__ candidate_neighbors,
    const rbmd::Id* __restrict__ candidate_num,
    rbmd::Id candidate_capacity, bool implicit_offsets,
    const rbmd::Id* __restrict__ max_core_neighbor_num,
    const rbmd::Id* __restrict__ core_start,
    rbmd::Id* __restrict__ core_end, rbmd::Id* __restrict__ core_neighbors,
    rbmd::Id* __restrict__ core_neighbor_num,
    rbmd::Id* __restrict__ required_core_neighbor_capacity, Box box,
    rbmd::Id* __restrict__ should_realloc) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  __shared__ typename WarpScan::TempStorage
      scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];
  const rbmd::Id group = threadIdx.x / THREADS_PER_PARTICLE;
  const rbmd::Id lane = threadIdx.x % THREADS_PER_PARTICLE;
  const rbmd::Id atom_idx =
      blockIdx.x * (blockDim.x / THREADS_PER_PARTICLE) + group;
  const bool active = atom_idx < total_atom_num;
  const rbmd::Real x = active ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real y = active ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real z = active ? __ldg(&pz[atom_idx]) : rbmd::Real(0);
  const std::uint64_t candidate_begin =
      active ? (implicit_offsets
                    ? static_cast<std::uint64_t>(atom_idx) *
                          static_cast<std::uint64_t>(candidate_capacity)
                    : static_cast<std::uint64_t>(
                          __ldg(&candidate_start[atom_idx])))
             : std::uint64_t(0);
  const rbmd::Id candidate_count =
      active ? (implicit_offsets
                    ? __ldg(&candidate_num[atom_idx])
                    : __ldg(&candidate_end[atom_idx]) -
                          __ldg(&candidate_start[atom_idx]))
             : rbmd::Id(0);
  const rbmd::Id output_start = active ? __ldg(&core_start[atom_idx]) : 0;
  const rbmd::Id output_capacity =
      active ? __ldg(&max_core_neighbor_num[atom_idx]) : 0;
  rbmd::Id written = 0;
  for (rbmd::Id base = 0; base < candidate_count;
       base += THREADS_PER_PARTICLE) {
    const rbmd::Id offset_in_row = base + lane;
    rbmd::Id neighbor_idx = 0;
    rbmd::Id accepted = 0;
    if (active && offset_in_row < candidate_count) {
      neighbor_idx = __ldg(&candidate_neighbors[
          candidate_begin + static_cast<std::uint64_t>(offset_in_row)]);
      if (neighbor_idx != atom_idx &&
          CaculateDistance(box, x, y, z, __ldg(&px[neighbor_idx]),
                           __ldg(&py[neighbor_idx]),
                           __ldg(&pz[neighbor_idx])) < core_cutoff_2) {
        accepted = 1;
      }
    }
    rbmd::Id compact_offset = 0;
    rbmd::Id accepted_in_group = 0;
    WarpScan(scan_storage[group])
        .ExclusiveSum(accepted, compact_offset, accepted_in_group);
    if (accepted && written + compact_offset < output_capacity) {
      core_neighbors[output_start + written + compact_offset] = neighbor_idx;
    }
    written += accepted_in_group;
  }
  if (active && lane == 0) {
    core_end[atom_idx] = output_start + MIN(written, output_capacity);
    core_neighbor_num[atom_idx] = written;
    rbmd::AtomicMaxCount(required_core_neighbor_capacity, written);
    if (written > output_capacity) {
      rbmd::AtomicOrCount(should_realloc, RBMD_TRUE);
    }
  }
}

template <int THREADS_PER_PARTICLE>
__global__ void SampleRBLNeighborCandidatesGrouped(
    rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* __restrict__ atoms_id,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const rbmd::Id* __restrict__ candidate_start,
    const rbmd::Id* __restrict__ candidate_end,
    const rbmd::Id* __restrict__ candidate_neighbors,
    const rbmd::Id* __restrict__ candidate_num,
    rbmd::Id candidate_capacity, bool implicit_offsets,
    rbmd::Id random_neighbor_capacity,
    rbmd::Id* __restrict__ random_neighbors,
    rbmd::Id* __restrict__ random_neighbors_num,
    rbmd::Real* __restrict__ random_neighbor_weight,
    rbmd::Id* __restrict__ required_random_neighbor_capacity,
    Box box, rbmd::Id selection_frequency, rbmd::Id sample_epoch) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  __shared__ typename WarpScan::TempStorage
      scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];

  const rbmd::Id group_in_block = threadIdx.x / THREADS_PER_PARTICLE;
  const rbmd::Id lane = threadIdx.x % THREADS_PER_PARTICLE;
  const rbmd::Id atom_idx =
      blockIdx.x * (blockDim.x / THREADS_PER_PARTICLE) + group_in_block;
  const bool active = atom_idx < total_atom_num;
  const std::uint64_t begin =
      active ? (implicit_offsets
                    ? static_cast<std::uint64_t>(atom_idx) *
                          static_cast<std::uint64_t>(candidate_capacity)
                    : static_cast<std::uint64_t>(
                          __ldg(&candidate_start[atom_idx])))
             : std::uint64_t(0);
  const rbmd::Id candidate_count =
      active ? (implicit_offsets
                    ? __ldg(&candidate_num[atom_idx])
                    : __ldg(&candidate_end[atom_idx]) -
                          __ldg(&candidate_start[atom_idx]))
             : rbmd::Id(0);

  const rbmd::Id frequency = selection_frequency > 0
                                 ? selection_frequency
                                 : rbmd::Id(1);
  const rbmd::Real center_x = active ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_y = active ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_z = active ? __ldg(&pz[atom_idx]) : rbmd::Real(0);
  const rbmd::Id atom_id = active ? __ldg(&atoms_id[atom_idx]) : 0;
  rbmd::Id sampled_count = 0;

  for (rbmd::Id base = 0; base < candidate_count;
       base += THREADS_PER_PARTICLE) {
    const rbmd::Id candidate_offset = base + lane;
    rbmd::Id neighbor_idx = 0;
    rbmd::Id selected = 0;
    if (active && candidate_offset < candidate_count) {
      neighbor_idx = __ldg(&candidate_neighbors[
          begin + static_cast<std::uint64_t>(candidate_offset)]);
      if (neighbor_idx != atom_idx) {
        rbmd::Real dx = center_x - __ldg(&px[neighbor_idx]);
        rbmd::Real dy = center_y - __ldg(&py[neighbor_idx]);
        rbmd::Real dz = center_z - __ldg(&pz[neighbor_idx]);
        MinImageDistance(box, dx, dy, dz);
        const rbmd::Real distance_2 = dx * dx + dy * dy + dz * dz;
        selected = distance_2 >= core_cutoff_2 &&
                           distance_2 < force_cutoff_2 &&
                           RblShellSampleSelected(
                               frequency, sample_epoch, atom_id,
                               __ldg(&atoms_id[neighbor_idx]))
                       ? rbmd::Id(1)
                       : rbmd::Id(0);
      }
    }

    rbmd::Id compact_offset = 0;
    rbmd::Id selected_in_group = 0;
    WarpScan(scan_storage[group_in_block])
        .ExclusiveSum(selected, compact_offset, selected_in_group);
    if (selected &&
        sampled_count + compact_offset < random_neighbor_capacity) {
      const unsigned long output =
          static_cast<unsigned long>(sampled_count + compact_offset) *
              static_cast<unsigned long>(total_atom_num) +
          static_cast<unsigned long>(atom_idx);
      random_neighbors[output] = neighbor_idx;
    }
    sampled_count += selected_in_group;
  }
  SYNCWARP(0xFFFFFFFF);

  if (active && lane == 0) {
    random_neighbors_num[atom_idx] =
        MIN(sampled_count, random_neighbor_capacity);
    random_neighbor_weight[atom_idx] = static_cast<rbmd::Real>(frequency);
    rbmd::AtomicMaxCount(required_random_neighbor_capacity, sampled_count);
  }
}

void GenerateRblCompactCandidatesOp<device::DEVICE_GPU>::Count(
    const rbmd::Id* per_atom_cell_id,
    const rbmd::Id* in_atom_list_start_index,
    const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
    rbmd::Id total_cells, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, const rbmd::Id* sorted_atom_indices,
    const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
    const rbmd::Real* sorted_pz,
    const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id* candidate_num, rbmd::Id* max_candidate_num, Box box,
    rbmd::Id* error_code) {
  constexpr int threads_per_particle = 8;
  const rbmd::Id atoms_per_block = BLOCK_SIZE / threads_per_particle;
  const unsigned int blocks_per_grid =
      (active_atom_num + atoms_per_block - 1) / atoms_per_block;
  CHECK_KERNEL((CountRblCompactCandidatesGrouped<threads_per_particle>
                <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
                    per_atom_cell_id, in_atom_list_start_index,
                    in_atom_list_end_index, cutoff_2, active_atom_num,
                    total_storage_atoms, total_cells, px, py, pz,
                    sorted_atom_indices, sorted_px, sorted_py, sorted_pz,
                    neighbor_cell, neighbor_cell_num,
                    candidate_num, max_candidate_num, box, error_code)));
}

void GenerateRblCompactCandidatesOp<device::DEVICE_GPU>::Append(
    const rbmd::Id* per_atom_cell_id,
    const rbmd::Id* in_atom_list_start_index,
    const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
    rbmd::Id total_cells, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, const rbmd::Id* sorted_atom_indices,
    const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
    const rbmd::Real* sorted_pz,
    const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id* candidate_num, rbmd::Id* candidate_start,
    rbmd::Id* candidate_end, rbmd::Id* candidates,
    std::uint64_t candidate_storage_size, rbmd::Id* total_candidate_num,
    rbmd::Id* max_candidate_num, Box box, rbmd::Id* error_code) {
  constexpr int threads_per_particle = 8;
  constexpr int max_candidates_per_atom = 96;
  const rbmd::Id atoms_per_block = BLOCK_SIZE / threads_per_particle;
  const unsigned int blocks_per_grid =
      (active_atom_num + atoms_per_block - 1) / atoms_per_block;
  CHECK_KERNEL((AppendRblCompactCandidatesGrouped<
                threads_per_particle, max_candidates_per_atom>
                <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
                    per_atom_cell_id, in_atom_list_start_index,
                    in_atom_list_end_index, cutoff_2, active_atom_num,
                    total_storage_atoms, total_cells, px, py, pz,
                    sorted_atom_indices, sorted_px, sorted_py, sorted_pz,
                    neighbor_cell, neighbor_cell_num,
                    candidate_num, candidate_start, candidate_end, candidates,
                    candidate_storage_size, total_candidate_num,
                    max_candidate_num, box, error_code)));
}

void GenerateRblCompactCandidatesOp<device::DEVICE_GPU>::Fill(
    const rbmd::Id* per_atom_cell_id,
    const rbmd::Id* in_atom_list_start_index,
    const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
    rbmd::Id total_cells, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, const rbmd::Id* sorted_atom_indices,
    const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
    const rbmd::Real* sorted_pz,
    const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
    const rbmd::Id* candidate_num, const rbmd::Id* candidate_start,
    rbmd::Id* candidate_end, rbmd::Id* candidates,
    std::uint64_t candidate_storage_size, Box box, rbmd::Id* error_code) {
  constexpr int threads_per_particle = 4;
  const rbmd::Id atoms_per_block = BLOCK_SIZE / threads_per_particle;
  const unsigned int blocks_per_grid =
      (active_atom_num + atoms_per_block - 1) / atoms_per_block;
  CHECK_KERNEL((FillRblCompactCandidatesGrouped<threads_per_particle>
                <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
                    per_atom_cell_id, in_atom_list_start_index,
                    in_atom_list_end_index, cutoff_2, active_atom_num,
                    total_storage_atoms, total_cells, px, py, pz,
                    sorted_atom_indices, sorted_px, sorted_py, sorted_pz,
                    neighbor_cell, neighbor_cell_num,
                    candidate_num, candidate_start, candidate_end, candidates,
                    candidate_storage_size, box, error_code)));
}

template <int THREADS_PER_PARTICLE>
void LaunchRblFixedCandidates(
    const rbmd::Id* per_atom_cell_id,
    const rbmd::Id* in_atom_list_start_index,
    const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Real core_cutoff_2,
    rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
    rbmd::Id total_cells, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, const rbmd::Id* sorted_atom_indices,
    const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
    const rbmd::Real* sorted_pz,
    const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id candidate_capacity, const rbmd::Id* candidate_start,
    rbmd::Id* candidate_end, rbmd::Id* candidates, rbmd::Id* candidate_num,
    bool implicit_offsets, std::uint64_t candidate_storage_size,
    rbmd::Id core_capacity, const rbmd::Id* core_start,
    rbmd::Id* core_end, rbmd::Id* core_neighbors,
    rbmd::Id* core_neighbor_num, std::uint64_t core_storage_size,
    Box box, rbmd::Id* status) {
  const rbmd::Id atoms_per_block = BLOCK_SIZE / THREADS_PER_PARTICLE;
  const unsigned int blocks =
      (active_atom_num + atoms_per_block - 1) / atoms_per_block;
  CHECK_KERNEL((FillRblFixedCandidatesGrouped<THREADS_PER_PARTICLE>
                <<<blocks, BLOCK_SIZE, 0, 0>>>(
                    per_atom_cell_id, in_atom_list_start_index,
                    in_atom_list_end_index, cutoff_2, core_cutoff_2,
                    active_atom_num,
                    total_storage_atoms, total_cells, px, py, pz,
                    sorted_atom_indices, sorted_px, sorted_py, sorted_pz,
                    neighbor_cell, neighbor_cell_num,
                    candidate_capacity, candidate_start, candidate_end,
                    candidates, candidate_num, implicit_offsets,
                    candidate_storage_size, core_capacity, core_start,
                    core_end, core_neighbors, core_neighbor_num,
                    core_storage_size, box, status)));
}

void GenerateRblFixedCandidatesOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id* per_atom_cell_id,
    const rbmd::Id* in_atom_list_start_index,
    const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Real core_cutoff_2,
    rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
    rbmd::Id total_cells, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, const rbmd::Id* sorted_atom_indices,
    const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
    const rbmd::Real* sorted_pz,
    const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id candidate_capacity, const rbmd::Id* candidate_start,
    rbmd::Id* candidate_end, rbmd::Id* candidates, rbmd::Id* candidate_num,
    bool implicit_offsets, std::uint64_t candidate_storage_size,
    rbmd::Id core_capacity, const rbmd::Id* core_start,
    rbmd::Id* core_end, rbmd::Id* core_neighbors,
    rbmd::Id* core_neighbor_num, std::uint64_t core_storage_size,
    Box box, rbmd::Id* status, rbmd::Id threads_per_particle) {
#define RBMD_LAUNCH_FIXED_CANDIDATES(TPP)                                   \
  LaunchRblFixedCandidates<TPP>(                                           \
      per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,  \
      cutoff_2, core_cutoff_2, active_atom_num, total_storage_atoms,        \
      total_cells, px, py, pz, sorted_atom_indices, sorted_px, sorted_py,   \
      sorted_pz, neighbor_cell, neighbor_cell_num, candidate_capacity,      \
      candidate_start, candidate_end, candidates, candidate_num,            \
      implicit_offsets, candidate_storage_size, core_capacity, core_start,  \
      core_end, core_neighbors, core_neighbor_num, core_storage_size, box,  \
      status)
  switch (threads_per_particle) {
    case 4:
      RBMD_LAUNCH_FIXED_CANDIDATES(4);
      break;
    case 16:
      RBMD_LAUNCH_FIXED_CANDIDATES(16);
      break;
    case 32:
      RBMD_LAUNCH_FIXED_CANDIDATES(32);
      break;
    case 64:
      RBMD_LAUNCH_FIXED_CANDIDATES(WARP_SIZE);
      break;
    case 8:
    default:
      RBMD_LAUNCH_FIXED_CANDIDATES(8);
      break;
  }
#undef RBMD_LAUNCH_FIXED_CANDIDATES
}

void FilterRblNeighborCandidatesOp<device::DEVICE_GPU>::operator()(
    rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* atoms_id, const rbmd::Real* px,
    const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Id* candidate_start, const rbmd::Id* candidate_end,
    const rbmd::Id* candidate_neighbors,
    const rbmd::Id* candidate_num, rbmd::Id candidate_capacity,
    bool implicit_offsets,
    const rbmd::Id* max_core_neighbor_num, const rbmd::Id* core_start,
    rbmd::Id* core_end, rbmd::Id* core_neighbors,
    rbmd::Id* core_neighbor_num, rbmd::Id random_neighbor_capacity,
    rbmd::Id* random_neighbors,
    rbmd::Id* random_neighbors_num,
    rbmd::Id* required_random_neighbor_capacity,
    rbmd::Id* required_core_neighbor_capacity, Box box,
    rbmd::Id* should_realloc,
    rbmd::Id selection_frequency, rbmd::Id sample_epoch,
    rbmd::Id threads_per_particle) {
  (void)force_cutoff_2;
  (void)atoms_id;
  (void)random_neighbor_capacity;
  (void)random_neighbors;
  (void)random_neighbors_num;
  (void)required_random_neighbor_capacity;
  (void)selection_frequency;
  (void)sample_epoch;
#define RBMD_LAUNCH_FILTER_CANDIDATES(TPP)                                \
  do {                                                                    \
    const rbmd::Id atoms_per_block = BLOCK_SIZE / TPP;                    \
    const unsigned int blocks_per_grid =                                  \
        (total_atom_num + atoms_per_block - 1) / atoms_per_block;         \
    CHECK_KERNEL((FilterRBLCoreCandidatesGrouped<TPP>                     \
                  <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(                \
                      core_cutoff_2, total_atom_num, px, py, pz,          \
                      candidate_start, candidate_end, candidate_neighbors,\
                      candidate_num, candidate_capacity, implicit_offsets,\
                      max_core_neighbor_num, core_start, core_end,        \
                      core_neighbors, core_neighbor_num,                  \
                      required_core_neighbor_capacity, box,              \
                      should_realloc)));                                  \
  } while (false)
  switch (threads_per_particle) {
    case 4:
      RBMD_LAUNCH_FILTER_CANDIDATES(4);
      break;
    case 16:
      RBMD_LAUNCH_FILTER_CANDIDATES(16);
      break;
    case 32:
      RBMD_LAUNCH_FILTER_CANDIDATES(32);
      break;
    case 64:
      RBMD_LAUNCH_FILTER_CANDIDATES(WARP_SIZE);
      break;
    case 8:
    default:
      RBMD_LAUNCH_FILTER_CANDIDATES(8);
      break;
  }
#undef RBMD_LAUNCH_FILTER_CANDIDATES
}

template <int THREADS_PER_PARTICLE>
void LaunchSampleRblNeighborCandidates(
    rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* atoms_id, const rbmd::Real* px,
    const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Id* candidate_start, const rbmd::Id* candidate_end,
    const rbmd::Id* candidate_neighbors, const rbmd::Id* candidate_num,
    rbmd::Id candidate_capacity, bool implicit_offsets,
    rbmd::Id random_neighbor_capacity,
    rbmd::Id* random_neighbors, rbmd::Id* random_neighbors_num,
    rbmd::Real* random_neighbor_weight,
    rbmd::Id* required_random_neighbor_capacity, Box box, rbmd::Id selection_frequency, rbmd::Id sample_epoch) {
  const rbmd::Id atoms_per_block = BLOCK_SIZE / THREADS_PER_PARTICLE;
  const unsigned int blocks_per_grid =
      (total_atom_num + atoms_per_block - 1) / atoms_per_block;
  CHECK_KERNEL((SampleRBLNeighborCandidatesGrouped<THREADS_PER_PARTICLE>
                <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
                    core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id,
                    px, py, pz, candidate_start, candidate_end,
                    candidate_neighbors, candidate_num, candidate_capacity,
                    implicit_offsets, random_neighbor_capacity,
                    random_neighbors, random_neighbors_num,
                    random_neighbor_weight, required_random_neighbor_capacity,
                    box, selection_frequency,
                    sample_epoch)));
}

void SampleRblNeighborCandidatesOp<device::DEVICE_GPU>::operator()(
    rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* atoms_id, const rbmd::Real* px,
    const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Id* candidate_start, const rbmd::Id* candidate_end,
    const rbmd::Id* candidate_neighbors, const rbmd::Id* candidate_num,
    rbmd::Id candidate_capacity, bool implicit_offsets,
    rbmd::Id random_neighbor_capacity,
    rbmd::Id* random_neighbors, rbmd::Id* random_neighbors_num,
    rbmd::Real* random_neighbor_weight,
    rbmd::Id* required_random_neighbor_capacity, Box box,
    rbmd::Id selection_frequency, rbmd::Id sample_epoch,
    rbmd::Id threads_per_particle) {
  switch (threads_per_particle) {
    case 1:
      LaunchSampleRblNeighborCandidates<1>(
          core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id, px, py, pz,
          candidate_start, candidate_end, candidate_neighbors,
          candidate_num, candidate_capacity, implicit_offsets,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          random_neighbor_weight, required_random_neighbor_capacity, box,
          selection_frequency, sample_epoch);
      break;
    case 4:
      LaunchSampleRblNeighborCandidates<4>(
          core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id, px, py, pz,
          candidate_start, candidate_end, candidate_neighbors,
          candidate_num, candidate_capacity, implicit_offsets,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          random_neighbor_weight, required_random_neighbor_capacity, box,
          selection_frequency, sample_epoch);
      break;
    case 16:
      LaunchSampleRblNeighborCandidates<16>(
          core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id, px, py, pz,
          candidate_start, candidate_end, candidate_neighbors,
          candidate_num, candidate_capacity, implicit_offsets,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          random_neighbor_weight, required_random_neighbor_capacity, box,
          selection_frequency, sample_epoch);
      break;
    case 32:
      LaunchSampleRblNeighborCandidates<32>(
          core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id, px, py, pz,
          candidate_start, candidate_end, candidate_neighbors,
          candidate_num, candidate_capacity, implicit_offsets,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          random_neighbor_weight, required_random_neighbor_capacity, box,
          selection_frequency, sample_epoch);
      break;
    case 64:
      LaunchSampleRblNeighborCandidates<WARP_SIZE>(
          core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id, px, py, pz,
          candidate_start, candidate_end, candidate_neighbors,
          candidate_num, candidate_capacity, implicit_offsets,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          random_neighbor_weight, required_random_neighbor_capacity, box,
          selection_frequency, sample_epoch);
      break;
    case 8:
    default:
      LaunchSampleRblNeighborCandidates<8>(
          core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id, px, py, pz,
          candidate_start, candidate_end, candidate_neighbors,
          candidate_num, candidate_capacity, implicit_offsets,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          random_neighbor_weight, required_random_neighbor_capacity, box,
          selection_frequency, sample_epoch);
      break;
  }
}

template <int THREADS_PER_PARTICLE>
void LaunchGenerateRblFullNeighborList(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real trunc_distance_power_2,
    rbmd::Real cutoff_2, rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
    rbmd::Real* px,
    rbmd::Real* py, rbmd::Real* pz, rbmd::Id* sorted_atom_indices,
    const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id* neighbor_num, rbmd::Id max_neighbor_num,
    rbmd::Id* neighbor_start, rbmd::Id* neighbor_end, rbmd::Id* neighbors,
    rbmd::Id random_neighbor_capacity, rbmd::Id* random_neighbors,
    rbmd::Id* random_neighbors_num,
    rbmd::Real* random_neighbor_weight,
    rbmd::Id* required_random_neighbor_capacity, Box box,
    rbmd::Id* should_realloc,
    rbmd::Id* required_neighbor_capacity, rbmd::Id selection_frequency,
    rbmd::Id sample_epoch) {
  const rbmd::Id atoms_per_block = BLOCK_SIZE / THREADS_PER_PARTICLE;
  const unsigned int blocks_per_grid =
      (total_atom_num + atoms_per_block - 1) / atoms_per_block;
  CHECK_KERNEL((GenerateRBLFullNeighborListGrouped<THREADS_PER_PARTICLE>)
                   <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          trunc_distance_power_2, cutoff_2, total_atom_num, atoms_id, px, py,
          pz, sorted_atom_indices, neighbor_cell, neighbor_cell_num,
          neighbor_num, max_neighbor_num, neighbor_start, neighbor_end,
          neighbors,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          random_neighbor_weight,
          required_random_neighbor_capacity, box, should_realloc,
          required_neighbor_capacity, selection_frequency, sample_epoch));
}

void GenerateRblFullNeighborListOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real trunc_distance_power_2,
    rbmd::Real cutoff_2, rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
    rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    rbmd::Id* sorted_atom_indices, const rbmd::Id* neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id* neighbor_num,
    rbmd::Id max_neighbor_num, rbmd::Id* neighbor_start,
    rbmd::Id* neighbor_end, rbmd::Id* neighbors,
    rbmd::Id random_neighbor_capacity, rbmd::Id* random_neighbors,
    rbmd::Id* random_neighbors_num, rbmd::Real* random_neighbor_weight,
    rbmd::Id* required_random_neighbor_capacity, Box box,
    rbmd::Id* should_realloc, rbmd::Id* required_neighbor_capacity,
    rbmd::Id selection_frequency, rbmd::Id sample_epoch,
    rbmd::Id threads_per_particle) {
#define RBMD_LAUNCH_RBL_DIRECT(TPP)                                      \
  LaunchGenerateRblFullNeighborList<TPP>(                               \
      per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,\
      trunc_distance_power_2, cutoff_2, total_atom_num, atoms_id, px, py,\
      pz, sorted_atom_indices, neighbor_cell, neighbor_cell_num,         \
      neighbor_num, max_neighbor_num, neighbor_start, neighbor_end,      \
      neighbors, random_neighbor_capacity, random_neighbors,             \
      random_neighbors_num, random_neighbor_weight,                      \
      required_random_neighbor_capacity, box, should_realloc,            \
      required_neighbor_capacity, selection_frequency, sample_epoch)
  switch (threads_per_particle) {
    case 4:
      RBMD_LAUNCH_RBL_DIRECT(4);
      break;
    case 16:
      RBMD_LAUNCH_RBL_DIRECT(16);
      break;
    case 32:
      RBMD_LAUNCH_RBL_DIRECT(32);
      break;
    case 64:
      RBMD_LAUNCH_RBL_DIRECT(WARP_SIZE);
      break;
    case 8:
    default:
      RBMD_LAUNCH_RBL_DIRECT(8);
      break;
  }
#undef RBMD_LAUNCH_RBL_DIRECT
}
namespace {
__device__ rbmd::Real FixedpDistance2(rbmd::Id owner, rbmd::Id other,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz, Box box, bool cvff_distance) {
  rbmd::Real dx=px[owner]-px[other], dy=py[owner]-py[other], dz=pz[owner]-pz[other];
  if(cvff_distance) MinImageDistance_fix(box,dx,dy,dz);
  else MinImageDistance(box,dx,dy,dz);
  return dx*dx+dy*dy+dz*dz;
}
__device__ void FixedpError(rbmd::Id* error, rbmd::Id code) {
  rbmd::AtomicOrCount(error,code);
}
// Validation only: compare the force INPUT, never classify a stored alias with
// another call-context norm at an exact cutoff. Consumer kernels remain unchanged.
struct FixedpForceInput { rbmd::Real x,y,z; };
__device__ FixedpForceInput FixedpCoreForceInput(rbmd::Id owner, rbmd::Id other,
    const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,Box box,bool cvff) {
  FixedpForceInput v{px[other]-px[owner],py[other]-py[owner],pz[other]-pz[owner]};
  if(cvff) MinImageDistance_fix(box,v.x,v.y,v.z);
  else MinImageDistance(box,v.x,v.y,v.z);
  return v;
}
__device__ rbmd::Id FixedpCoreAliasInputKind(rbmd::Id owner, rbmd::Id representative,
    rbmd::Id alias,const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
    Box box,bool cvff) {
  if(alias==representative) return 1; // same memory input; no numerical reclassification
  if(px[alias]==px[representative] && py[alias]==py[representative] &&
      pz[alias]==pz[representative]) return 2;
  const auto a=FixedpCoreForceInput(owner,representative,px,py,pz,box,cvff);
  const auto b=FixedpCoreForceInput(owner,alias,px,py,pz,box,cvff);
  return a.x==b.x && a.y==b.y && a.z==b.z ? rbmd::Id(3) : rbmd::Id(0);
}
__global__ void PrepareRblFixedpRows(rbmd::Id owned, rbmd::Id storage_atoms,
    const rbmd::Id* atom_ids, const rbmd::Id* atom_types,
    const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, const rbmd::Id* candidate_start,
    const rbmd::Id* candidate_end, rbmd::Id* candidates,
    const rbmd::Id* candidate_num, rbmd::Id candidate_capacity,
    bool implicit_offsets, std::uint64_t candidate_storage_size,
    const rbmd::Id* core_start, const rbmd::Id* core_end,
    const rbmd::Id* core_neighbors, std::uint64_t core_storage_size,
    rbmd::Real core_superset_2, rbmd::Real physical_core_2, Box box, bool cvff_distance, rbmd::Id* error) {
  const rbmd::Id i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=owned) return;
  const rbmd::Id num=implicit_offsets?candidate_num[i]:candidate_end[i]-candidate_start[i];
  const std::uint64_t begin=implicit_offsets?std::uint64_t(i)*std::uint64_t(candidate_capacity):
      (candidate_start[i]>=0?std::uint64_t(candidate_start[i]):candidate_storage_size);
  if(num<0 || begin>candidate_storage_size || std::uint64_t(num)>candidate_storage_size-begin ||
      (implicit_offsets && num>candidate_capacity)) {FixedpError(error,1);return;}
  for(rbmd::Id j=0;j<num;++j) {
    const rbmd::Id other=candidates[begin+std::uint64_t(j)];
    if(other<0 || other>=storage_atoms) {FixedpError(error,2);return;}
  }
  // Keep ALL aliases. No arbitrary alias is dropped during cached-row rebuild.
  rbmd::fixedp::SortRow(candidates+begin,std::uint64_t(num),atom_ids);
  for(rbmd::Id j=1;j<num;++j) {
    const rbmd::Id a=candidates[begin+std::uint64_t(j-1)],b=candidates[begin+std::uint64_t(j)];
    if(atom_ids[a]==atom_ids[b] && (atom_types[a]!=atom_types[b] || charge[a]!=charge[b])) {
      FixedpError(error,4);return;
    }
  }
  const rbmd::Id cb=core_start[i],ce=core_end[i];
  if(cb<0 || ce<cb || std::uint64_t(ce)>core_storage_size) {FixedpError(error,8);return;}
  // Validate the unchanged core+skin list only at a REAL rebuild, not each force call.
  for(rbmd::Id j=cb;j<ce;++j) {
    const rbmd::Id other=core_neighbors[j];
    if(other<0 || other>=storage_atoms) {FixedpError(error,16);return;}
    if(atom_ids[other]==atom_ids[i]) continue;
    for(rbmd::Id k=cb;k<j;++k)
      if(atom_ids[core_neighbors[k]]==atom_ids[other]) {FixedpError(error,32);return;}
    bool in_full=false;
    for(rbmd::Id k=0;k<num;++k)
      if(atom_ids[candidates[begin+std::uint64_t(k)]]==atom_ids[other]) {in_full=true;break;}
    if(!in_full) {FixedpError(error,64);return;}
  }
  for(rbmd::Id j=0;j<num;) {
    const rbmd::Id first=candidates[begin+std::uint64_t(j)],gid=atom_ids[first];
    // V4 separates the cache-construction and force-coverage contracts.
    // Core+skin was constructed by CaculateDistance, not the force-style norm.
    rbmd::Real cache_nearest=CaculateDistance(box,px[i],py[i],pz[i],px[first],py[first],pz[first]);
    rbmd::Real physical_nearest=FixedpDistance2(i,first,px,py,pz,box,cvff_distance);
    rbmd::Id physical_alias=first,next=j+1;
    while(next<num && atom_ids[candidates[begin+std::uint64_t(next)]]==gid) {
      const rbmd::Id alias=candidates[begin+std::uint64_t(next)];
      const rbmd::Real cache_d2=CaculateDistance(box,px[i],py[i],pz[i],px[alias],py[alias],pz[alias]);
      const rbmd::Real physical_d2=FixedpDistance2(i,alias,px,py,pz,box,cvff_distance);
      if(cache_d2<cache_nearest) cache_nearest=cache_d2;
      if(physical_d2<physical_nearest || (physical_d2==physical_nearest && alias<physical_alias)) {
        physical_nearest=physical_d2;physical_alias=alias;
      }
      ++next;
    }
    if(gid!=atom_ids[i]) {
      const bool cache_required=cache_nearest<core_superset_2;
      const bool physical_required=physical_nearest>EPSILON && physical_nearest<=physical_core_2;
      bool in_core=false,valid_physical_alias=false;
      if(cache_required || physical_required) {
        for(rbmd::Id k=cb;k<ce;++k) {
          const rbmd::Id alias=core_neighbors[k];
          if(atom_ids[alias]!=gid) continue;
          in_core=true;
          if(atom_types[alias]!=atom_types[physical_alias] || charge[alias]!=charge[physical_alias]) {
            FixedpError(error,4);return;
          }
          if(FixedpCoreAliasInputKind(i,physical_alias,alias,px,py,pz,box,cvff_distance))
            valid_physical_alias=true;
        }
      }
      // Physical representative must be present as the SAME force input, not merely the same GID.
      // A separate norm evaluation cannot overrule identical inputs at an FP32 boundary.
      if(physical_required && !valid_physical_alias) {FixedpError(error,2048);return;}
      if(cache_required && !in_core) {FixedpError(error,128);return;}
    }
    j=next;
  }
}
__global__ void SampleRblFixedpRows(rbmd::Real core_2, rbmd::Real outer_2,
    rbmd::Id owned, rbmd::Id storage_atoms, const rbmd::Id* atom_ids,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Id* candidate_start, const rbmd::Id* candidate_end,
    const rbmd::Id* candidates, const rbmd::Id* candidate_num,
    rbmd::Id candidate_capacity, bool implicit_offsets,
    rbmd::Id requested, rbmd::Id capacity, rbmd::Id* output,
    rbmd::Id* selected_num, rbmd::Id* shell_population,
    rbmd::Real* weight, rbmd::Id* required_capacity, Box box, bool cvff_distance, rbmd::Id epoch, rbmd::Id* error) {
  const rbmd::Id i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=owned) return;
  selected_num[i]=0;shell_population[i]=0;weight[i]=rbmd::Real(0);
  if(requested<=0 || capacity<requested) {FixedpError(error,256);return;}
  const std::uint64_t begin=implicit_offsets?std::uint64_t(i)*std::uint64_t(candidate_capacity):
      std::uint64_t(candidate_start[i]);
  const rbmd::Id num=implicit_offsets?candidate_num[i]:candidate_end[i]-candidate_start[i];
  rbmd::fixedp::Counter rng{std::uint64_t(atom_ids[i]),std::uint64_t(epoch)};
  rbmd::Id population=0;
  for(rbmd::Id j=0;j<num;) {
    rbmd::Id representative=candidates[begin+std::uint64_t(j)];
    const rbmd::Id gid=atom_ids[representative];
    rbmd::Real nearest=FixedpDistance2(i,representative,px,py,pz,box,cvff_distance);
    if(!(nearest>=rbmd::Real(0)) || nearest>std::numeric_limits<rbmd::Real>::max()) {FixedpError(error,512);return;}
    rbmd::Id next=j+1;
    while(next<num && atom_ids[candidates[begin+std::uint64_t(next)]]==gid) {
      const rbmd::Id alias=candidates[begin+std::uint64_t(next)];
      const rbmd::Real d2=FixedpDistance2(i,alias,px,py,pz,box,cvff_distance);
      if(!(d2>=rbmd::Real(0)) || d2>std::numeric_limits<rbmd::Real>::max()) {FixedpError(error,512);return;}
      if(d2<nearest || (d2==nearest && alias<representative)) {nearest=d2;representative=alias;}
      ++next;
    }
    // Same strict physical squared-distance predicate as both current force kernels.
    // core, skin-only, self/ghost-self and duplicate IDs never increment M.
    if(gid!=atom_ids[i] && nearest>core_2 && nearest<outer_2) {
      ++population;
      if(!rbmd::fixedp::ReservoirAccept(representative,std::uint64_t(population),
          requested,output,std::uint64_t(owned),std::uint64_t(i),rng)) {
        FixedpError(error,1024);return;
      }
    }
    j=next;
  }
  const rbmd::Id k=population<requested?population:requested;
  shell_population[i]=population;selected_num[i]=k;
  rbmd::AtomicMaxCount(required_capacity,k);
  weight[i]=k>0?rbmd::Real(population)/rbmd::Real(k):rbmd::Real(0);
}

// This runs only AFTER the original guard has failed. All original failures remain failures.
__global__ void DiagnoseRblFixedpMissingCore(rbmd::Id owned, rbmd::Id storage_atoms,
    const rbmd::Id* atom_ids, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, const rbmd::Id* candidate_start,
    const rbmd::Id* candidate_end, const rbmd::Id* candidates,
    const rbmd::Id* candidate_num, rbmd::Id candidate_capacity,
    bool implicit_offsets, std::uint64_t candidate_storage_size,
    const rbmd::Id* core_start, const rbmd::Id* core_end,
    const rbmd::Id* core_neighbors, std::uint64_t core_storage_size,
    rbmd::Real threshold, rbmd::Real physical_core_2, Box box, bool cvff_distance, unsigned int* claimed,
    rbmd::Id* wid, rbmd::Real* val) {
  const rbmd::Id i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=owned) return;
  const rbmd::Id num=implicit_offsets?candidate_num[i]:candidate_end[i]-candidate_start[i];
  const std::uint64_t begin=implicit_offsets?std::uint64_t(i)*std::uint64_t(candidate_capacity):
      (candidate_start[i]>=0?std::uint64_t(candidate_start[i]):candidate_storage_size);
  if(num<0 || begin>candidate_storage_size || std::uint64_t(num)>candidate_storage_size-begin ||
      (implicit_offsets && num>candidate_capacity)) return;
  const rbmd::Id cb=core_start[i],ce=core_end[i];
  if(cb<0 || ce<cb || std::uint64_t(ce)>core_storage_size) return;
  for(rbmd::Id j=0;j<num;) {
    const rbmd::Id first=candidates[begin+std::uint64_t(j)];
    if(first<0 || first>=storage_atoms) return;
    const rbmd::Id gid=atom_ids[first];
    rbmd::Real force_min=FixedpDistance2(i,first,px,py,pz,box,cvff_distance);
    rbmd::Real filter_min=CaculateDistance(box,px[i],py[i],pz[i],px[first],py[first],pz[first]);
    rbmd::Id force_alias=first,filter_alias=first,next=j+1;
    while(next<num) {
      const rbmd::Id a=candidates[begin+std::uint64_t(next)];
      if(a<0 || a>=storage_atoms) return;
      if(atom_ids[a]!=gid) break;
      const rbmd::Real f=FixedpDistance2(i,a,px,py,pz,box,cvff_distance);
      const rbmd::Real c=CaculateDistance(box,px[i],py[i],pz[i],px[a],py[a],pz[a]);
      if(f<force_min){force_min=f;force_alias=a;}
      if(c<filter_min){filter_min=c;filter_alias=a;}
      ++next;
    }
    if(gid!=atom_ids[i] && (force_min<threshold || filter_min<threshold)) {
      bool found=false,physical_found=false;
      rbmd::Id stored_alias=-1,stored_input_kind=0;
      rbmd::Real core_alias_min=std::numeric_limits<rbmd::Real>::max();
      for(rbmd::Id k=cb;k<ce;++k){const rbmd::Id a=core_neighbors[k];
        if(a<0 || a>=storage_atoms) return;
        if(atom_ids[a]==gid){found=true;
          const rbmd::Real d2=FixedpDistance2(i,a,px,py,pz,box,cvff_distance);
          const rbmd::Id input_kind=FixedpCoreAliasInputKind(i,force_alias,a,px,py,pz,box,cvff_distance);
          if(stored_alias<0 || d2<core_alias_min){core_alias_min=d2;stored_alias=a;stored_input_kind=input_kind;}
          if(input_kind)physical_found=true;}}
      const bool cache_missing=filter_min<threshold && !found;
      const bool physical_missing=force_min>EPSILON && force_min<=physical_core_2 && !physical_found;
      if((cache_missing || physical_missing) && atomicCAS(claimed,0u,1u)==0u) {
        wid[0]=i;wid[1]=atom_ids[i];wid[2]=gid;wid[3]=force_alias;wid[4]=filter_alias;
        wid[5]=cb;wid[6]=ce;wid[7]=num;wid[8]=next-j;wid[9]=filter_min<threshold?1:0;
        wid[18]=stored_alias;wid[19]=stored_input_kind;
        val[0]=force_min;val[1]=filter_min;val[2]=threshold;val[26]=physical_core_2;val[27]=core_alias_min;
        const auto candidate_input=FixedpCoreForceInput(i,force_alias,px,py,pz,box,cvff_distance);
        val[31]=candidate_input.x;val[32]=candidate_input.y;val[33]=candidate_input.z;
        if(stored_alias>=0){
          val[28]=px[stored_alias];val[29]=py[stored_alias];val[30]=pz[stored_alias];
          const auto stored_input=FixedpCoreForceInput(i,stored_alias,px,py,pz,box,cvff_distance);
          val[34]=stored_input.x;val[35]=stored_input.y;val[36]=stored_input.z;
        }
        val[3]=px[i];val[4]=py[i];val[5]=pz[i];
        val[6]=px[force_alias];val[7]=py[force_alias];val[8]=pz[force_alias];
        val[9]=px[filter_alias];val[10]=py[filter_alias];val[11]=pz[filter_alias];
        for(int a=0;a<3;++a){val[12+a]=box._length[a];val[15+a]=box._length_inv[a];}
        for(rbmd::Id k=cb;k<ce && k-cb<8;++k){const rbmd::Id a=core_neighbors[k];
          wid[10+(k-cb)]=atom_ids[a];
          val[18+(k-cb)]=CaculateDistance(box,px[i],py[i],pz[i],px[a],py[a],pz[a]);}
        return;
      }
    }
    j=next;
  }
}

}  // anonymous namespace

void DiagnoseRblFixedpMissingCoreOp<device::DEVICE_GPU>::operator()(
    rbmd::Id owned, rbmd::Id storage_atoms, const rbmd::Id* atom_ids,
    const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
    const rbmd::Id* candidate_start,const rbmd::Id* candidate_end,
    const rbmd::Id* candidates,const rbmd::Id* candidate_num,
    rbmd::Id candidate_capacity,bool implicit_offsets,std::uint64_t candidate_storage_size,
    const rbmd::Id* core_start,const rbmd::Id* core_end,const rbmd::Id* core_neighbors,
    std::uint64_t core_storage_size,rbmd::Real threshold,rbmd::Real physical_core_2,Box box,bool cvff_distance,
    unsigned int* claimed,rbmd::Id* wid,rbmd::Real* val) {
  if(owned<=0) return;
  const unsigned int blocks=(owned+BLOCK_SIZE-1)/BLOCK_SIZE;
  CHECK_KERNEL((DiagnoseRblFixedpMissingCore<<<blocks,BLOCK_SIZE,0,0>>>(owned,storage_atoms,
      atom_ids,px,py,pz,candidate_start,candidate_end,candidates,candidate_num,candidate_capacity,
      implicit_offsets,candidate_storage_size,core_start,core_end,core_neighbors,core_storage_size,
      threshold,physical_core_2,box,cvff_distance,claimed,wid,val)));
}

void PrepareRblFixedpRowsOp<device::DEVICE_GPU>::operator()(rbmd::Id owned,
    rbmd::Id storage_atoms, const rbmd::Id* atom_ids, const rbmd::Id* atom_types,
    const rbmd::Real* charge, const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
    const rbmd::Id* candidate_start,const rbmd::Id* candidate_end,rbmd::Id* candidates,
    const rbmd::Id* candidate_num,rbmd::Id candidate_capacity,bool implicit_offsets,
    std::uint64_t candidate_storage_size,const rbmd::Id* core_start,const rbmd::Id* core_end,
    const rbmd::Id* core_neighbors,std::uint64_t core_storage_size,
    rbmd::Real core_superset_2,rbmd::Real physical_core_2,Box box,bool cvff_distance,rbmd::Id* error) {
  if(owned<=0) return;
  const unsigned int blocks=(owned+BLOCK_SIZE-1)/BLOCK_SIZE;
  CHECK_KERNEL((PrepareRblFixedpRows<<<blocks,BLOCK_SIZE,0,0>>>(owned,storage_atoms,
      atom_ids,atom_types,charge,px,py,pz,candidate_start,candidate_end,candidates,
      candidate_num,candidate_capacity,implicit_offsets,candidate_storage_size,
      core_start,core_end,core_neighbors,core_storage_size,core_superset_2,physical_core_2,box,cvff_distance,error)));
}
void SampleRblFixedpRowsOp<device::DEVICE_GPU>::operator()(rbmd::Real core_2,
    rbmd::Real outer_2,rbmd::Id owned,rbmd::Id storage_atoms,const rbmd::Id* atom_ids,
    const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
    const rbmd::Id* candidate_start,const rbmd::Id* candidate_end,const rbmd::Id* candidates,
    const rbmd::Id* candidate_num,rbmd::Id candidate_capacity,bool implicit_offsets,
    rbmd::Id requested,rbmd::Id capacity,rbmd::Id* output,rbmd::Id* selected_num,
    rbmd::Id* population,rbmd::Real* weight,rbmd::Id* required_capacity,Box box,bool cvff_distance,rbmd::Id epoch,rbmd::Id* error) {
  if(owned<=0) return;
  const unsigned int blocks=(owned+BLOCK_SIZE-1)/BLOCK_SIZE;
  CHECK_KERNEL((SampleRblFixedpRows<<<blocks,BLOCK_SIZE,0,0>>>(core_2,outer_2,
      owned,storage_atoms,atom_ids,px,py,pz,candidate_start,candidate_end,candidates,
      candidate_num,candidate_capacity,implicit_offsets,requested,capacity,output,
      selected_num,population,weight,required_capacity,box,cvff_distance,epoch,error)));
}

}  // namespace op
