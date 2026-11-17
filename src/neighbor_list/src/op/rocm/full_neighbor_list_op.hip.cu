#include "linked_cell_locator.h"
#include "common/atomic_count.h"
#include "common/device_types.h"
#include "common/rbmd_define.h"
#include "common/types.h"
#include "model/box.h"
#include "src/op/full_neighbor_list_op.h"
#include <cstdint>

namespace op {
__device__ rbmd::Id WrapFullNeighborStencilIndex(long long value,
                                                 rbmd::Id width) {
  const long long signed_width = static_cast<long long>(width);
  long long wrapped = value % signed_width;
  if (wrapped < 0) {
    wrapped += signed_width;
  }
  return static_cast<rbmd::Id>(wrapped);
}

__device__ rbmd::Id ComputeFullNeighborStencilCell(
    const LinkedCellDeviceDataPtr* linked_cell, rbmd::Id cell_idx,
    rbmd::Id stencil_idx, rbmd::Id cell_count_within_cutoff) {
  const rbmd::Id dim_x = linked_cell->_d_per_dimension_cells[0];
  const rbmd::Id dim_y = linked_cell->_d_per_dimension_cells[1];
  const rbmd::Id dim_z = linked_cell->_d_per_dimension_cells[2];
  const rbmd::Id stencil_width = 2 * cell_count_within_cutoff + 1;
  const rbmd::Id span_x =
      MIN(stencil_width, linked_cell->_d_covers_whole_domain[0]
                             ? linked_cell->_d_nativate_cells_dims[0]
                             : dim_x);
  const rbmd::Id span_y =
      MIN(stencil_width, linked_cell->_d_covers_whole_domain[1]
                             ? linked_cell->_d_nativate_cells_dims[1]
                             : dim_y);

  const long long offset_x =
      static_cast<long long>(stencil_idx % span_x) -
      static_cast<long long>(cell_count_within_cutoff);
  const long long offset_y =
      static_cast<long long>((stencil_idx / span_x) % span_y) -
      static_cast<long long>(cell_count_within_cutoff);
  const long long offset_z =
      static_cast<long long>(stencil_idx / (span_x * span_y)) -
      static_cast<long long>(cell_count_within_cutoff);

  const rbmd::Id cell_z = cell_idx / (dim_x * dim_y);
  const rbmd::Id cell_y = (cell_idx - cell_z * dim_x * dim_y) / dim_x;
  const rbmd::Id cell_x = cell_idx - dim_x * (cell_y + dim_y * cell_z);

  rbmd::Id neighbor_x = 0;
  rbmd::Id neighbor_y = 0;
  rbmd::Id neighbor_z = 0;

  if (linked_cell->_d_covers_whole_domain[0]) {
    const rbmd::Id native_width = linked_cell->_d_nativate_cells_dims[0];
    const long long native_idx =
        static_cast<long long>(cell_x) -
        static_cast<long long>(linked_cell->_d_nativate_cells_start[0]);
    neighbor_x =
        WrapFullNeighborStencilIndex(native_idx + offset_x, native_width) +
                 linked_cell->_d_nativate_cells_start[0];
  } else {
    neighbor_x = WrapFullNeighborStencilIndex(
        static_cast<long long>(cell_x) + offset_x, dim_x);
  }

  if (linked_cell->_d_covers_whole_domain[1]) {
    const rbmd::Id native_width = linked_cell->_d_nativate_cells_dims[1];
    const long long native_idx =
        static_cast<long long>(cell_y) -
        static_cast<long long>(linked_cell->_d_nativate_cells_start[1]);
    neighbor_y =
        WrapFullNeighborStencilIndex(native_idx + offset_y, native_width) +
                 linked_cell->_d_nativate_cells_start[1];
  } else {
    neighbor_y = WrapFullNeighborStencilIndex(
        static_cast<long long>(cell_y) + offset_y, dim_y);
  }

  if (linked_cell->_d_covers_whole_domain[2]) {
    const rbmd::Id native_width = linked_cell->_d_nativate_cells_dims[2];
    const long long native_idx =
        static_cast<long long>(cell_z) -
        static_cast<long long>(linked_cell->_d_nativate_cells_start[2]);
    neighbor_z =
        WrapFullNeighborStencilIndex(native_idx + offset_z, native_width) +
                 linked_cell->_d_nativate_cells_start[2];
  } else {
    neighbor_z = WrapFullNeighborStencilIndex(
        static_cast<long long>(cell_z) + offset_z, dim_z);
  }

  return (neighbor_z * dim_y + neighbor_y) * dim_x + neighbor_x;
}

__global__ void ComputeFullNeighbors(
    LinkedCellDeviceDataPtr* linked_cell,
    rbmd::Id* neighbor_cell,
    Cell* cells,
    rbmd::Id neighbor_num,
    rbmd::Id cell_count_within_cutoff,
    const rbmd::Id* __restrict__ stencil_indices
) {
    const unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell_idx < linked_cell->_d_total_cells && !cells[cell_idx]._is_halo) {
        const std::uint64_t row_offset =
            static_cast<std::uint64_t>(cell_idx) *
            static_cast<std::uint64_t>(neighbor_num);
        for (rbmd::Id stencil_idx = 0; stencil_idx < neighbor_num;
             ++stencil_idx) {
          neighbor_cell[row_offset + static_cast<std::uint64_t>(stencil_idx)] =
              ComputeFullNeighborStencilCell(
                                             linked_cell, cell_idx,
                                             stencil_indices[stencil_idx],
                                             cell_count_within_cutoff);
        }
    }
}

// use while total_cell < computed neighbor_num
__global__ void ComputeFullNeighborsWithoutPBC(rbmd::Id* neighbor_cell,
                                               rbmd::Id neighbor_num,
                                               rbmd::Id total_cell) {
  const unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_idx < total_cell) {
    const std::uint64_t row_offset =
        static_cast<std::uint64_t>(cell_idx) *
        static_cast<std::uint64_t>(neighbor_num);
    int neighbor_count = 0;
    for (int neighbor_cell_idx = 0; neighbor_cell_idx < neighbor_num;
         ++neighbor_cell_idx) {
      neighbor_cell[row_offset +
                    static_cast<std::uint64_t>(neighbor_count)] =
          neighbor_cell_idx;
      ++neighbor_count;
         }
  }
}

template <int THREADS_PER_PARTICLE>
__global__ void GenerateFullNeighborListWarp(
    rbmd::Id* __restrict__ per_atom_cell_id,
    rbmd::Id* __restrict__ in_atom_list_start_index,
    rbmd::Id* __restrict__ in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, rbmd::Real* __restrict__ px,
    rbmd::Real* __restrict__ py, rbmd::Real* __restrict__ pz,
    rbmd::Id* __restrict__ sorted_atom_indices,
    const rbmd::Id* __restrict__ neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id* __restrict__ neighbor_num,
    rbmd::Id max_neighbor_num,
    rbmd::Id* __restrict__ neighbor_start, rbmd::Id* __restrict__ neighbor_end,
    rbmd::Id* __restrict__ neighbors, Box box,
    rbmd::Id* __restrict__ should_realloc,
    rbmd::Id* __restrict__ required_neighbor_capacity) {
  using WarpScan = WARPSCAN<rbmd::Id, THREADS_PER_PARTICLE>;
  __shared__ typename WarpScan::TempStorage
      scan_storage[BLOCK_SIZE / THREADS_PER_PARTICLE];

  const rbmd::Id group_in_block = threadIdx.x / THREADS_PER_PARTICLE;
  const rbmd::Id lane = threadIdx.x % THREADS_PER_PARTICLE;
  const rbmd::Id atom_idx =
      blockIdx.x * (blockDim.x / THREADS_PER_PARTICLE) + group_in_block;
  const bool active = atom_idx < total_atom_num;
  const rbmd::Id cell_idx = active ? __ldg(&per_atom_cell_id[atom_idx]) : 0;
  const std::uint64_t neighbor_cell_row =
      active ? static_cast<std::uint64_t>(cell_idx) *
                   static_cast<std::uint64_t>(neighbor_cell_num)
             : std::uint64_t(0);
  const rbmd::Real center_x = active ? __ldg(&px[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_y = active ? __ldg(&py[atom_idx]) : rbmd::Real(0);
  const rbmd::Real center_z = active ? __ldg(&pz[atom_idx]) : rbmd::Real(0);
  const rbmd::Id my_start = active ? neighbor_start[atom_idx] : 0;
  rbmd::Id nneigh = 0;

  for (rbmd::Id cell_offset = 0; cell_offset < neighbor_cell_num;
       ++cell_offset) {
    rbmd::Id start = 0;
    rbmd::Id end = 0;
    if (active) {
      const rbmd::Id neighbour_cell_idx = __ldg(
          &neighbor_cell[neighbor_cell_row +
                         static_cast<std::uint64_t>(cell_offset)]);
      start = __ldg(&in_atom_list_start_index[neighbour_cell_idx]);
      end = __ldg(&in_atom_list_end_index[neighbour_cell_idx]);
    }

    // 同一 logical group 内所有 lane 参与相同次数的 subgroup scan。
    // 不可改用 wave-wide ballot：同一 wave 中的其他 particle group 可能因
    // cell_size 不同而处于分歧控制流，ballot 会把邻居计数和写入偏移算错。
    const rbmd::Id cell_size = end > start ? end - start : 0;
    for (rbmd::Id base = 0; base < cell_size;
         base += THREADS_PER_PARTICLE) {
      const rbmd::Id pos = start + base + lane;
      rbmd::Id candidate = 0;
      rbmd::Id has_neighbor = 0;
      if (active && pos < end) {
        const rbmd::Id neighbor_atom_idx =
            __ldg(&sorted_atom_indices[pos]);
        if (neighbor_atom_idx != atom_idx &&
            CaculateDistance(box, center_x, center_y, center_z,
                             __ldg(&px[neighbor_atom_idx]),
                             __ldg(&py[neighbor_atom_idx]),
                             __ldg(&pz[neighbor_atom_idx])) < cutoff_2) {
          candidate = neighbor_atom_idx;
          has_neighbor = 1;
        }
      }

      rbmd::Id compact_offset = 0;
      rbmd::Id accepted_in_group = 0;
      WarpScan(scan_storage[group_in_block])
          .ExclusiveSum(has_neighbor, compact_offset, accepted_in_group);

      if (has_neighbor && (nneigh + compact_offset) < max_neighbor_num) {
        neighbors[my_start + nneigh + compact_offset] = candidate;
      }
      nneigh += accepted_in_group;
    }
    SYNCWARP(0xFFFFFFFF);
  }

  if (active && lane == 0) {
    neighbor_num[atom_idx] = nneigh;
    neighbor_end[atom_idx] = my_start + MIN(nneigh, max_neighbor_num);
    rbmd::AtomicMaxCount(required_neighbor_capacity, nneigh);
    if (nneigh > max_neighbor_num) {
      rbmd::AtomicOrCount(should_realloc, RBMD_TRUE);
    }
  }
}

void ComputeFullNeighborsOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* neighbor_cell,
    Cell* cells,
    rbmd::Id neighbor_num,
    rbmd::Id total_cell,
    rbmd::Id cell_count_within_cutoff,
    const rbmd::Id* stencil_indices
) {
  // 直接获取 LinkedCell 的设备数据指针
  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto linked_cell_ptr = linked_cell->GetDataPtr();

  unsigned int blocks_per_grid = (total_cell + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(ComputeFullNeighbors<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      linked_cell_ptr,  // 直接传递结构体指针
      neighbor_cell,
      cells,
      neighbor_num,
      cell_count_within_cutoff,
      stencil_indices
  ));
}

void ComputeFullNeighborsWithoutPBCOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* neighbor_cell, rbmd::Id neighbor_num, rbmd::Id total_cell) {
  unsigned int blocks_per_grid = (total_cell + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(
      ComputeFullNeighborsWithoutPBC<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          neighbor_cell, neighbor_num, total_cell));
}

template <int THREADS_PER_PARTICLE>
void LaunchGenerateFullNeighborList(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    rbmd::Id* sorted_atom_indices, const rbmd::Id* neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id* neighbor_num,
    rbmd::Id max_neighbor_num, rbmd::Id* neighbor_start,
    rbmd::Id* neighbor_end, rbmd::Id* neighbors, Box box,
    rbmd::Id* should_realloc, rbmd::Id* required_neighbor_capacity) {
  const rbmd::Id atoms_per_block = BLOCK_SIZE / THREADS_PER_PARTICLE;
  const unsigned int blocks_per_grid =
      (total_atom_num + atoms_per_block - 1) / atoms_per_block;
  CHECK_KERNEL((GenerateFullNeighborListWarp<THREADS_PER_PARTICLE>
                <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
                    per_atom_cell_id, in_atom_list_start_index,
                    in_atom_list_end_index, cutoff_2, total_atom_num, px, py,
                    pz, sorted_atom_indices, neighbor_cell, neighbor_cell_num,
                    neighbor_num, max_neighbor_num, neighbor_start,
                    neighbor_end, neighbors, box, should_realloc,
                    required_neighbor_capacity)));
}

void GenerateFullNeighborListOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    rbmd::Id* sorted_atom_indices, const rbmd::Id* neighbor_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id* neighbor_num,
    rbmd::Id max_neighbor_num, rbmd::Id* neighbor_start,
    rbmd::Id* neighbor_end, rbmd::Id* neighbors, Box box,
    rbmd::Id* should_realloc, rbmd::Id* required_neighbor_capacity,
    rbmd::Id threads_per_particle) {
  switch (threads_per_particle) {
    case 1:
      LaunchGenerateFullNeighborList<1>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          neighbor_cell, neighbor_cell_num, neighbor_num, max_neighbor_num,
          neighbor_start, neighbor_end, neighbors, box, should_realloc,
          required_neighbor_capacity);
      break;
    case 4:
      LaunchGenerateFullNeighborList<4>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          neighbor_cell, neighbor_cell_num, neighbor_num, max_neighbor_num,
          neighbor_start, neighbor_end, neighbors, box, should_realloc,
          required_neighbor_capacity);
      break;
    case 8:
      LaunchGenerateFullNeighborList<8>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          neighbor_cell, neighbor_cell_num, neighbor_num, max_neighbor_num,
          neighbor_start, neighbor_end, neighbors, box, should_realloc,
          required_neighbor_capacity);
      break;
    case 16:
      LaunchGenerateFullNeighborList<16>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          neighbor_cell, neighbor_cell_num, neighbor_num, max_neighbor_num,
          neighbor_start, neighbor_end, neighbors, box, should_realloc,
          required_neighbor_capacity);
      break;
    case 32:
      LaunchGenerateFullNeighborList<32>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          neighbor_cell, neighbor_cell_num, neighbor_num, max_neighbor_num,
          neighbor_start, neighbor_end, neighbors, box, should_realloc,
          required_neighbor_capacity);
      break;
    default:
      LaunchGenerateFullNeighborList<WARP_SIZE>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          neighbor_cell, neighbor_cell_num, neighbor_num, max_neighbor_num,
          neighbor_start, neighbor_end, neighbors, box, should_realloc,
          required_neighbor_capacity);
      break;
  }
}
}  // namespace op
