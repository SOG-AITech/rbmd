#include "full_neighbor_list_op.h"
// #include "hiprand/hiprand_kernel.h" TODO
#include "rbl_full_neighbor_list_op.h"
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

__device__ rbmd::Id WrapStencilIndex(long long value, rbmd::Id width) {
  const long long signed_width = static_cast<long long>(width);
  long long wrapped = value % signed_width;
  if (wrapped < 0) {
    wrapped += signed_width;
  }
  return static_cast<rbmd::Id>(wrapped);
}

__device__ rbmd::Id ComputeFullStencilCellOnTheFly(
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
    neighbor_x = WrapStencilIndex(native_idx + offset_x, native_width) +
                 linked_cell->_d_nativate_cells_start[0];
  } else {
    neighbor_x =
        WrapStencilIndex(static_cast<long long>(cell_x) + offset_x, dim_x);
  }

  if (linked_cell->_d_covers_whole_domain[1]) {
    const rbmd::Id native_width = linked_cell->_d_nativate_cells_dims[1];
    const long long native_idx =
        static_cast<long long>(cell_y) -
        static_cast<long long>(linked_cell->_d_nativate_cells_start[1]);
    neighbor_y = WrapStencilIndex(native_idx + offset_y, native_width) +
                 linked_cell->_d_nativate_cells_start[1];
  } else {
    neighbor_y =
        WrapStencilIndex(static_cast<long long>(cell_y) + offset_y, dim_y);
  }

  if (linked_cell->_d_covers_whole_domain[2]) {
    const rbmd::Id native_width = linked_cell->_d_nativate_cells_dims[2];
    const long long native_idx =
        static_cast<long long>(cell_z) -
        static_cast<long long>(linked_cell->_d_nativate_cells_start[2]);
    neighbor_z = WrapStencilIndex(native_idx + offset_z, native_width) +
                 linked_cell->_d_nativate_cells_start[2];
  } else {
    neighbor_z =
        WrapStencilIndex(static_cast<long long>(cell_z) + offset_z, dim_z);
  }

  return (neighbor_z * dim_y + neighbor_y) * dim_x + neighbor_x;
}

__global__ void EstimateRBLFullNeighborList(
    rbmd::Id* __restrict__ per_atom_cell_id,
    rbmd::Id* __restrict__ in_atom_list_start_index,
    rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real trunc_distance_power_2, rbmd::Id total_atom_num,
    rbmd::Real* __restrict__ px, rbmd::Real* __restrict__ py,
    rbmd::Real* __restrict__ pz,
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

  for (rbmd::Id i = 0; i < neighbor_cell_num; ++i) {
    const rbmd::Id neighbour_cell_idx = ComputeFullStencilCellOnTheFly(
        linked_cell, cell_idx, i, cell_count_within_cutoff);
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
          box, __ldg(&px[atom_idx]), __ldg(&py[atom_idx]),
          __ldg(&pz[atom_idx]), __ldg(&px[neighbor_atom_idx]),
          __ldg(&py[neighbor_atom_idx]), __ldg(&pz[neighbor_atom_idx]));
      if (distance < trunc_distance_power_2) {
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

// TODO warp op and optimize
//! Note 暂时未测试Warp的
__global__ void GenerateRBLFullNeighborList(
    rbmd::Id* __restrict__ per_atom_cell_id,
    rbmd::Id* __restrict__ in_atom_list_start_index,
    rbmd::Id* __restrict__ in_atom_list_end_index,
    rbmd::Real trunc_distance_power_2, rbmd::Real cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* __restrict__ atoms_id,
    rbmd::Real* __restrict__ px,
    rbmd::Real* __restrict__ py, rbmd::Real* __restrict__ pz,
    rbmd::Id* __restrict__ sorted_atom_indices,
    rbmd::Id* __restrict__ max_neighbor_num,
    rbmd::Id* __restrict__ neighbor_start, rbmd::Id* __restrict__ neighbor_end,
    rbmd::Id* __restrict__ neighbors, rbmd::Id random_neighbor_capacity,
    rbmd::Id* __restrict__ random_neighbors,
    rbmd::Id* __restrict__ random_neighbors_num,
    rbmd::Id* __restrict__ required_random_neighbor_capacity, Box box,
    rbmd::Id* __restrict__ should_realloc, LinkedCellDeviceDataPtr* linked_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id selection_frequency,
    rbmd::Id cell_count_within_cutoff, rbmd::Id sample_epoch) {
  const unsigned int atom_idx = (blockIdx.x * blockDim.x + threadIdx.x);
  if (atom_idx < total_atom_num) {
    rbmd::Id neighbor_num = neighbor_start[atom_idx];
    rbmd::Id random_neighbor_num = 0;
    rbmd::Id total_neighbor_num = 0;
    rbmd::Id cell_idx = __ldg(&per_atom_cell_id[atom_idx]);
    const rbmd::Real center_x = __ldg(&px[atom_idx]);
    const rbmd::Real center_y = __ldg(&py[atom_idx]);
    const rbmd::Real center_z = __ldg(&pz[atom_idx]);
    for (rbmd::Id i = 0; i < neighbor_cell_num; ++i) {
      rbmd::Id neighbour_cell_idx =
          ComputeFullStencilCellOnTheFly(linked_cell, cell_idx, i,
                                         cell_count_within_cutoff);
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
            if (total_neighbor_num < max_neighbor_num[atom_idx]) {
              neighbors[neighbor_num] = neighbor_atom_idx;
              ++neighbor_num;
            } else {
              atomicOr(should_realloc, RBMD_TRUE);
            }
            ++total_neighbor_num;
          } else if (distance >= trunc_distance_power_2 &&
                     distance < cutoff_2) {
            if (RblShellSampleSelected(selection_frequency, sample_epoch,
                                       __ldg(&atoms_id[atom_idx]),
                                       __ldg(&atoms_id[neighbor_atom_idx]))) {
              if (random_neighbor_num < random_neighbor_capacity) {
                const unsigned long random_neighbor_index =
                    static_cast<unsigned long>(random_neighbor_num) *
                        static_cast<unsigned long>(total_atom_num) +
                    static_cast<unsigned long>(atom_idx);
                random_neighbors[random_neighbor_index] = neighbor_atom_idx;
              }
              random_neighbor_num++;
            }
          }
        }
      }
    }
    neighbor_end[atom_idx] = neighbor_num;
    random_neighbors_num[atom_idx] =
        MIN(random_neighbor_num, random_neighbor_capacity);
    atomicMax(required_random_neighbor_capacity, random_neighbor_num);
    rbmd::Id my_total_neighbor_num = total_neighbor_num;
    if (my_total_neighbor_num > max_neighbor_num[atom_idx]) {
      atomicOr(should_realloc, RBMD_TRUE);
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
    const rbmd::Id* __restrict__ max_core_neighbor_num,
    const rbmd::Id* __restrict__ core_start,
    rbmd::Id* __restrict__ core_end, rbmd::Id* __restrict__ core_neighbors,
    rbmd::Id random_neighbor_capacity,
    rbmd::Id* __restrict__ random_neighbors,
    rbmd::Id* __restrict__ random_neighbors_num,
    rbmd::Id* __restrict__ required_random_neighbor_capacity, Box box,
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

  for (rbmd::Id pos = candidate_start[atom_idx];
       pos < candidate_end[atom_idx]; ++pos) {
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
        atomicOr(should_realloc, RBMD_TRUE);
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
  random_neighbors_num[atom_idx] =
      MIN(sampled_count, random_neighbor_capacity);
  atomicMax(required_random_neighbor_capacity, sampled_count);
}

void FilterRblNeighborCandidatesOp<device::DEVICE_GPU>::operator()(
    rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
    rbmd::Id total_atom_num, const rbmd::Id* atoms_id, const rbmd::Real* px,
    const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Id* candidate_start, const rbmd::Id* candidate_end,
    const rbmd::Id* candidate_neighbors,
    const rbmd::Id* max_core_neighbor_num, const rbmd::Id* core_start,
    rbmd::Id* core_end, rbmd::Id* core_neighbors,
    rbmd::Id random_neighbor_capacity, rbmd::Id* random_neighbors,
    rbmd::Id* random_neighbors_num,
    rbmd::Id* required_random_neighbor_capacity, Box box,
    rbmd::Id* should_realloc,
    rbmd::Id selection_frequency, rbmd::Id sample_epoch) {
  const unsigned int blocks_per_grid =
      (total_atom_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(FilterRBLNeighborCandidates<<<blocks_per_grid, BLOCK_SIZE, 0,
                                             0>>>(
      core_cutoff_2, force_cutoff_2, total_atom_num, atoms_id, px, py, pz,
      candidate_start, candidate_end, candidate_neighbors,
      max_core_neighbor_num, core_start, core_end, core_neighbors,
      random_neighbor_capacity, random_neighbors, random_neighbors_num,
      required_random_neighbor_capacity, box, should_realloc,
      selection_frequency, sample_epoch));
}

void EstimateRblFullNeighborListOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real trunc_distance_power_2,
    rbmd::Id total_atom_num, rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    rbmd::Id* sorted_atom_indices, rbmd::Id* neighbour_num,
    rbmd::Id* max_neighbour_num, Box box, LinkedCellDeviceDataPtr* linked_cell,
    rbmd::Id neighbor_cell_num, rbmd::Id cell_count_within_cutoff) {
  unsigned int blocks_per_grid = (total_atom_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(EstimateRBLFullNeighborList<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
      trunc_distance_power_2, total_atom_num, px, py, pz, sorted_atom_indices,
      neighbour_num, max_neighbour_num, box, linked_cell, neighbor_cell_num,
      cell_count_within_cutoff));
}

void GenerateRblFullNeighborListOp<device::DEVICE_GPU>::operator()(
    rbmd::Id* per_atom_cell_id, rbmd::Id* in_atom_list_start_index,
    rbmd::Id* in_atom_list_end_index, rbmd::Real trunc_distance_power_2,
    rbmd::Real cutoff_2, rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
    rbmd::Real* px,
    rbmd::Real* py, rbmd::Real* pz, rbmd::Id* sorted_atom_indices,
    rbmd::Id* max_neighbor_num,
    rbmd::Id* neighbor_start, rbmd::Id* neighbor_end, rbmd::Id* neighbors,
    rbmd::Id random_neighbor_capacity, rbmd::Id* random_neighbors,
    rbmd::Id* random_neighbors_num,
    rbmd::Id* required_random_neighbor_capacity, Box box,
    rbmd::Id* should_realloc,
    LinkedCellDeviceDataPtr* linked_cell, rbmd::Id neighbor_cell_num,
    rbmd::Id selection_frequency, rbmd::Id cell_count_within_cutoff,
    rbmd::Id sample_epoch) {
  unsigned int blocks_per_grid = (total_atom_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(
      GenerateRBLFullNeighborList<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          per_atom_cell_id, in_atom_list_start_index, in_atom_list_end_index,
          trunc_distance_power_2, cutoff_2, total_atom_num, atoms_id, px, py,
          pz,
          sorted_atom_indices,
          max_neighbor_num, neighbor_start, neighbor_end, neighbors,
          random_neighbor_capacity, random_neighbors, random_neighbors_num,
          required_random_neighbor_capacity, box, should_realloc, linked_cell,
          neighbor_cell_num, selection_frequency, cell_count_within_cutoff,
          sample_epoch));
}
}  // namespace op
