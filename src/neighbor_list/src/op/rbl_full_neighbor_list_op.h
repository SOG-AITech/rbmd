#pragma once
#include "common/device_types.h"
#include "common/types.h"
#include "linked_cell/linked_cell.h"
#include "model/box.h"
namespace op {
template <typename DEVICE>
struct EstimateRblFullNeighborListOp {
  void operator()(rbmd::Id* per_atom_cell_id,
                  rbmd::Id* in_atom_list_start_index,
                  rbmd::Id* in_atom_list_end_index,
                  rbmd::Real trunc_distance_power_2,
                  rbmd::Id total_atom_num, rbmd::Real* px, rbmd::Real* py,
                  rbmd::Real* pz, rbmd::Id* sorted_atom_indices,
                  rbmd::Id* neighbour_num,
                  rbmd::Id* max_neighbour_num, Box box,
                  LinkedCellDeviceDataPtr* linked_cell,
                  rbmd::Id neighbor_cell_num,
                  rbmd::Id cell_count_within_cutoff);
};

template <>
struct EstimateRblFullNeighborListOp<device::DEVICE_GPU> {
  void operator()(rbmd::Id* per_atom_cell_id,
                  rbmd::Id* in_atom_list_start_index,
                  rbmd::Id* in_atom_list_end_index,
                  rbmd::Real trunc_distance_power_2,
                  rbmd::Id total_atom_num, rbmd::Real* px, rbmd::Real* py,
                  rbmd::Real* pz, rbmd::Id* sorted_atom_indices,
                  rbmd::Id* neighbour_num,
                  rbmd::Id* max_neighbour_num, Box box,
                  LinkedCellDeviceDataPtr* linked_cell,
                  rbmd::Id neighbor_cell_num,
                  rbmd::Id cell_count_within_cutoff);
};

template <typename DEVICE>
struct GenerateRblFullNeighborListOp {
  void operator()(rbmd::Id* per_atom_cell_id,
                  rbmd::Id* in_atom_list_start_index,
                  rbmd::Id* in_atom_list_end_index,
                  rbmd::Real trunc_distance_power_2, rbmd::Real cutoff_2,
                  rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
                  rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  rbmd::Id* sorted_atom_indices,
                  rbmd::Id* max_neighbor_num,
                  rbmd::Id* neighbor_start, rbmd::Id* neighbor_end,
                  rbmd::Id* neighbors, rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors, rbmd::Id* random_neighbors_num,
                  rbmd::Id* required_random_neighbor_capacity,
                  Box box, rbmd::Id* should_realloc,
                  LinkedCellDeviceDataPtr* linked_cell,
                  rbmd::Id neighbor_cell_num, rbmd::Id selection_frequency,
                  rbmd::Id cell_count_within_cutoff,
                  rbmd::Id sample_epoch);
};

template <>
struct GenerateRblFullNeighborListOp<device::DEVICE_GPU> {
  void operator()(rbmd::Id* per_atom_cell_id,
                  rbmd::Id* in_atom_list_start_index,
                  rbmd::Id* in_atom_list_end_index,
                  rbmd::Real trunc_distance_power_2, rbmd::Real cutoff_2,
                  rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
                  rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  rbmd::Id* sorted_atom_indices,
                  rbmd::Id* max_neighbor_num,
                  rbmd::Id* neighbor_start, rbmd::Id* neighbor_end,
                  rbmd::Id* neighbors, rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors, rbmd::Id* random_neighbors_num,
                  rbmd::Id* required_random_neighbor_capacity,
                  Box box, rbmd::Id* should_realloc,
                  LinkedCellDeviceDataPtr* linked_cell,
                  rbmd::Id neighbor_cell_num, rbmd::Id selection_frequency,
                  rbmd::Id cell_count_within_cutoff,
                  rbmd::Id sample_epoch);
};

template <typename DEVICE>
struct FilterRblNeighborCandidatesOp {
  void operator()(rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
                  rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz,
                  const rbmd::Id* candidate_start,
                  const rbmd::Id* candidate_end,
                  const rbmd::Id* candidate_neighbors,
                  const rbmd::Id* max_core_neighbor_num,
                  const rbmd::Id* core_start, rbmd::Id* core_end,
                  rbmd::Id* core_neighbors, rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors,
                  rbmd::Id* random_neighbors_num,
                  rbmd::Id* required_random_neighbor_capacity, Box box,
                  rbmd::Id* should_realloc, rbmd::Id selection_frequency,
                  rbmd::Id sample_epoch);
};

template <>
struct FilterRblNeighborCandidatesOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
                  rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz,
                  const rbmd::Id* candidate_start,
                  const rbmd::Id* candidate_end,
                  const rbmd::Id* candidate_neighbors,
                  const rbmd::Id* max_core_neighbor_num,
                  const rbmd::Id* core_start, rbmd::Id* core_end,
                  rbmd::Id* core_neighbors, rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors,
                  rbmd::Id* random_neighbors_num,
                  rbmd::Id* required_random_neighbor_capacity, Box box,
                  rbmd::Id* should_realloc, rbmd::Id selection_frequency,
                  rbmd::Id sample_epoch);
};

}  // namespace op
