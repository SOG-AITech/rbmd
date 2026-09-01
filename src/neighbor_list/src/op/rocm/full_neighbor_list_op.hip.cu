#include "linked_cell_locator.h"
#include "common/device_types.h"
#include "common/rbmd_define.h"
#include "common/types.h"
#include "model/box.h"
#include "src/op/full_neighbor_list_op.h"

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

__device__ rbmd::Id ComputeFullNeighborStencilCellOnTheFly(
    const LinkedCellDeviceDataPtr* linked_cell, rbmd::Id cell_idx,
    rbmd::Id stencil_idx, rbmd::Id cell_count_within_cutoff) {
  const rbmd::Id dim_x = linked_cell->_d_per_dimension_cells[0];
  const rbmd::Id dim_y = linked_cell->_d_per_dimension_cells[1];
  const rbmd::Id dim_z = linked_cell->_d_per_dimension_cells[2];
  const rbmd::Id stencil_width = 2 * cell_count_within_cutoff + 1;

  const long long offset_x =
      static_cast<long long>(stencil_idx % stencil_width) -
      static_cast<long long>(cell_count_within_cutoff);
  const long long offset_y =
      static_cast<long long>((stencil_idx / stencil_width) % stencil_width) -
      static_cast<long long>(cell_count_within_cutoff);
  const long long offset_z =
      static_cast<long long>(stencil_idx / (stencil_width * stencil_width)) -
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
    rbmd::Id cell_count_within_cutoff
) {
    const unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell_idx < linked_cell->_d_total_cells && !cells[cell_idx]._is_halo) {
        // 计算当前 cell 的 3D 索引
        rbmd::Id idx_z = cell_idx / (linked_cell->_d_per_dimension_cells[0] *
                                     linked_cell->_d_per_dimension_cells[1]);
        rbmd::Id idx_y = (cell_idx - idx_z * linked_cell->_d_per_dimension_cells[0] *
                         linked_cell->_d_per_dimension_cells[1]) /
                         linked_cell->_d_per_dimension_cells[0];
        rbmd::Id idx_x = cell_idx - linked_cell->_d_per_dimension_cells[0] *
                        (idx_y + linked_cell->_d_per_dimension_cells[1] * idx_z);

        int neighbor_count = 0;

        for (int dz = -cell_count_within_cutoff; dz <= cell_count_within_cutoff; ++dz) {
            for (int dy = -cell_count_within_cutoff; dy <= cell_count_within_cutoff; ++dy) {
                for (int dx = -cell_count_within_cutoff; dx <= cell_count_within_cutoff; ++dx) {

                    int neighbor_x, neighbor_y, neighbor_z;
                    // X 维度处理
                    if (linked_cell->_d_covers_whole_domain[0]) {
                        // 全域覆盖：只在 native cells 内使用 PBC
                        int native_width = linked_cell->_d_nativate_cells_dims[0];
                        int native_idx_x = idx_x - linked_cell->_d_nativate_cells_start[0];
                        int native_neighbor_x = (native_idx_x + dx + native_width) % native_width;
                        neighbor_x = native_neighbor_x + linked_cell->_d_nativate_cells_start[0];
                    } else {
                        // 正常 PBC，包括 halo cells
                        neighbor_x = (idx_x + dx + linked_cell->_d_per_dimension_cells[0]) %
                                    linked_cell->_d_per_dimension_cells[0];
                    }

                    // Y 维度处理
                    if (linked_cell->_d_covers_whole_domain[1]) {
                        int native_width = linked_cell->_d_nativate_cells_dims[1];
                        int native_idx_y = idx_y - linked_cell->_d_nativate_cells_start[1];
                        int native_neighbor_y = (native_idx_y + dy + native_width) % native_width;
                        neighbor_y = native_neighbor_y + linked_cell->_d_nativate_cells_start[1];
                    } else {
                        neighbor_y = (idx_y + dy + linked_cell->_d_per_dimension_cells[1]) %
                                    linked_cell->_d_per_dimension_cells[1];
                    }

                    // Z 维度处理
                    if (linked_cell->_d_covers_whole_domain[2]) {
                        int native_width = linked_cell->_d_nativate_cells_dims[2];
                        int native_idx_z = idx_z - linked_cell->_d_nativate_cells_start[2];
                        int native_neighbor_z = (native_idx_z + dz + native_width) % native_width;
                        neighbor_z = native_neighbor_z + linked_cell->_d_nativate_cells_start[2];
                    } else {
                        neighbor_z = (idx_z + dz + linked_cell->_d_per_dimension_cells[2]) %
                                    linked_cell->_d_per_dimension_cells[2];
                    }

                    // 计算邻居 cell 索引
                    rbmd::Id neighbour_cell_idx =
                        (neighbor_z * linked_cell->_d_per_dimension_cells[1] + neighbor_y) *
                        linked_cell->_d_per_dimension_cells[0] + neighbor_x;

                    neighbor_cell[cell_idx * neighbor_num + neighbor_count] = neighbour_cell_idx;
                    ++neighbor_count;
                }
            }
        }
    }
}

// use while total_cell < computed neighbor_num
__global__ void ComputeFullNeighborsWithoutPBC(rbmd::Id* neighbor_cell,
                                               rbmd::Id neighbor_num,
                                               rbmd::Id total_cell) {
  const unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_idx < total_cell) {
    int neighbor_count = 0;
    for (int neighbor_cell_idx = 0; neighbor_cell_idx < neighbor_num;
         ++neighbor_cell_idx) {
      neighbor_cell[cell_idx * neighbor_num + neighbor_count] =
          neighbor_cell_idx;
      ++neighbor_count;
         }
  }
}

__global__ void EstimateFullNeighborListNoWarp(
    rbmd::Id* __restrict__ per_atom_cell_id,
    rbmd::Id* __restrict__ in_atom_list_start_index,
    rbmd::Id* __restrict__ in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, rbmd::Real* __restrict__ px,
    rbmd::Real* __restrict__ py, rbmd::Real* __restrict__ pz,
    rbmd::Id* __restrict__ sorted_atom_indices,
    rbmd::Id* __restrict__ neighbour_num,
    rbmd::Id* __restrict__ max_neighbour_num, Box box,
    LinkedCellDeviceDataPtr* linked_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id cell_count_within_cutoff) {
  const unsigned int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom_idx >= total_atom_num) {
    return;
  }

  const rbmd::Id cell_idx = __ldg(&per_atom_cell_id[atom_idx]);
  rbmd::Id atom_neighbor_num = 0;

  for (int i = 0; i < neighbor_cell_num; ++i) {
    const rbmd::Id neighbour_cell_idx =
        ComputeFullNeighborStencilCellOnTheFly(linked_cell, cell_idx, i,
                                               cell_count_within_cutoff);
    const rbmd::Id start = __ldg(&in_atom_list_start_index[neighbour_cell_idx]);
    const rbmd::Id end = __ldg(&in_atom_list_end_index[neighbour_cell_idx]);

    for (rbmd::Id neighbor_atom_pos = start; neighbor_atom_pos < end;
         ++neighbor_atom_pos) {
      const rbmd::Id neighbor_atom_idx =
          __ldg(&sorted_atom_indices[neighbor_atom_pos]);
      if (atom_idx == neighbor_atom_idx) {
        continue;
      }
      const rbmd::Real distance = CaculateDistance(
          box, __ldg(&px[atom_idx]), __ldg(&py[atom_idx]), __ldg(&pz[atom_idx]),
          __ldg(&px[neighbor_atom_idx]), __ldg(&py[neighbor_atom_idx]),
          __ldg(&pz[neighbor_atom_idx]));
      if (distance < cutoff_2) {
        ++atom_neighbor_num;
      }
    }
  }

  neighbour_num[atom_idx] = atom_neighbor_num;
  max_neighbour_num[atom_idx] =
      MAX(((rbmd::Id)CEIL(atom_neighbor_num * SAFE_ZONE) + warpSize - 1) /
              warpSize * warpSize,
          MIN_NBNUM);
}

__global__ void GenerateFullNeighborListNoWarp(
    rbmd::Id* __restrict__ per_atom_cell_id,
    rbmd::Id* __restrict__ in_atom_list_start_index,
    rbmd::Id* __restrict__ in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, rbmd::Real* __restrict__ px,
    rbmd::Real* __restrict__ py, rbmd::Real* __restrict__ pz,
    rbmd::Id* __restrict__ sorted_atom_indices,
    rbmd::Id* __restrict__ max_neighbor_num,
    rbmd::Id* __restrict__ neighbor_start, rbmd::Id* __restrict__ neighbor_end,
    rbmd::Id* __restrict__ neighbors, Box  box,
    rbmd::Id* __restrict__ should_realloc, LinkedCellDeviceDataPtr* linked_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id cell_count_within_cutoff) {
  // cutoff2是平方
  // 计算当前线程处理的原子的索引    ---- 当前线程原子就是atom_idx
  const unsigned int atom_idx = (blockIdx.x * blockDim.x + threadIdx.x);
  rbmd::Real distance = 0;
  rbmd::Id neighbor_num = 0;
  if (atom_idx < total_atom_num) {
    neighbor_num = neighbor_start[atom_idx];  // 变量名不太好，这里没全部变为0
    rbmd::Id total_neighbor_num = 0;
    // 当前所属的cell
    rbmd::Id cell_idx =  __ldg(&per_atom_cell_id[atom_idx]);
    const rbmd::Real center_x = __ldg(&px[atom_idx]);
    const rbmd::Real center_y = __ldg(&py[atom_idx]);
    const rbmd::Real center_z = __ldg(&pz[atom_idx]);

    // 遍历当前原子的所有邻居cell
    for (rbmd::Id i = 0; i < neighbor_cell_num; ++i) {
      rbmd::Id neighbour_cell_idx =
          ComputeFullNeighborStencilCellOnTheFly(linked_cell, cell_idx, i,
                                                 cell_count_within_cutoff);
      rbmd::Id start = __ldg(&in_atom_list_start_index[neighbour_cell_idx]);
      rbmd::Id end = __ldg(&in_atom_list_end_index[neighbour_cell_idx]);
      for (rbmd::Id neighbor_atom_pos = start; neighbor_atom_pos < end;
           ++neighbor_atom_pos) {
        const rbmd::Id neighbor_atom_idx =
            __ldg(&sorted_atom_indices[neighbor_atom_pos]);
        // TODO 应该是不会有负数的
        if (atom_idx != neighbor_atom_idx) {
          distance = CaculateDistance(
              box, center_x, center_y, center_z,
              __ldg(&px[neighbor_atom_idx]),
              __ldg(&py[neighbor_atom_idx]), __ldg(&pz[neighbor_atom_idx]));
          if (distance < cutoff_2) {
            if (total_neighbor_num < max_neighbor_num[atom_idx]) {
              neighbors[neighbor_num] = neighbor_atom_idx;
              ++neighbor_num;
            } else {
              atomicOr(should_realloc, RBMD_TRUE);
            }
            ++total_neighbor_num;
          }
        }
           }
    }
    neighbor_end[atom_idx] = neighbor_num;
    rbmd::Id my_total_neighbor_num = total_neighbor_num;
    if (my_total_neighbor_num > max_neighbor_num[atom_idx]) {
      atomicOr(should_realloc, RBMD_TRUE);
    }
  }
}

void ComputeFullNeighborsOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* neighbor_cell,
    Cell* cells,
    rbmd::Id neighbor_num,
    rbmd::Id total_cell,
    rbmd::Id cell_count_within_cutoff
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
      cell_count_within_cutoff
  ));
}

void ComputeFullNeighborsWithoutPBCOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* neighbor_cell, rbmd::Id neighbor_num, rbmd::Id total_cell) {
  unsigned int blocks_per_grid = (total_cell + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(
      ComputeFullNeighborsWithoutPBC<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          neighbor_cell, neighbor_num, total_cell));
}

void EstimateFullNeighborListOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    rbmd::Id* sorted_atom_indices, rbmd::Id* neighbour_num,
    rbmd::Id* max_neighbour_num, Box box,
    LinkedCellDeviceDataPtr* linked_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id cell_count_within_cutoff) {
  unsigned int blocks_per_grid =
      (total_atom_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(
      EstimateFullNeighborListNoWarp<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          neighbour_num, max_neighbour_num, box, linked_cell,
          neighbor_cell_num, cell_count_within_cutoff));
}

void GenerateFullNeighborListOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    rbmd::Id* sorted_atom_indices, rbmd::Id* max_neighbor_num,
    rbmd::Id* neighbor_start,
    rbmd::Id* neighbor_end, rbmd::Id* neighbors, Box box,
    rbmd::Id* should_realloc, LinkedCellDeviceDataPtr* linked_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id cell_count_within_cutoff) {
  unsigned int blocks_per_grid =
      (total_atom_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(
      GenerateFullNeighborListNoWarp<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          cutoff_2, total_atom_num, px, py, pz, sorted_atom_indices,
          max_neighbor_num, neighbor_start, neighbor_end, neighbors, box,
          should_realloc, linked_cell, neighbor_cell_num,
          cell_count_within_cutoff));
}
}  // namespace op
