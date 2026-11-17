#pragma once
#include <cstdint>
#include "common/device_types.h"
#include "common/types.h"
#include "linked_cell/linked_cell.h"
#include "model/box.h"
namespace op {
template <typename DEVICE>
struct GenerateRblFullNeighborListOp {
  void operator()(rbmd::Id* per_atom_cell_id,
                  rbmd::Id* in_atom_list_start_index,
                  rbmd::Id* in_atom_list_end_index,
                  rbmd::Real trunc_distance_power_2, rbmd::Real cutoff_2,
                  rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
                  rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  rbmd::Id* sorted_atom_indices,
                  const rbmd::Id* neighbor_cell,
                  rbmd::Id neighbor_cell_num, rbmd::Id* neighbor_num,
                  rbmd::Id max_neighbor_num,
                  rbmd::Id* neighbor_start, rbmd::Id* neighbor_end,
                  rbmd::Id* neighbors, rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors, rbmd::Id* random_neighbors_num,
                  rbmd::Real* random_neighbor_weight,
                  rbmd::Id* required_random_neighbor_capacity,
                  Box box, rbmd::Id* should_realloc,
                  rbmd::Id* required_neighbor_capacity,
                  rbmd::Id selection_frequency, rbmd::Id sample_epoch,
                  rbmd::Id threads_per_particle);
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
                  const rbmd::Id* neighbor_cell,
                  rbmd::Id neighbor_cell_num, rbmd::Id* neighbor_num,
                  rbmd::Id max_neighbor_num,
                  rbmd::Id* neighbor_start, rbmd::Id* neighbor_end,
                  rbmd::Id* neighbors, rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors, rbmd::Id* random_neighbors_num,
                  rbmd::Real* random_neighbor_weight,
                  rbmd::Id* required_random_neighbor_capacity,
                  Box box, rbmd::Id* should_realloc,
                  rbmd::Id* required_neighbor_capacity,
                  rbmd::Id selection_frequency, rbmd::Id sample_epoch,
                  rbmd::Id threads_per_particle);
};

template <typename DEVICE>
struct GenerateRblCompactCandidatesOp {
  void Count(const rbmd::Id* per_atom_cell_id,
             const rbmd::Id* in_atom_list_start_index,
             const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
             rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
             rbmd::Id total_cells, const rbmd::Real* px,
             const rbmd::Real* py, const rbmd::Real* pz,
             const rbmd::Id* sorted_atom_indices,
             const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
             const rbmd::Real* sorted_pz,
             const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
             rbmd::Id* candidate_num, rbmd::Id* max_candidate_num,
             Box box, rbmd::Id* error_code);

  void Append(const rbmd::Id* per_atom_cell_id,
              const rbmd::Id* in_atom_list_start_index,
              const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
              rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
              rbmd::Id total_cells, const rbmd::Real* px,
              const rbmd::Real* py, const rbmd::Real* pz,
              const rbmd::Id* sorted_atom_indices,
              const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
              const rbmd::Real* sorted_pz,
              const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
              rbmd::Id* candidate_num, rbmd::Id* candidate_start,
              rbmd::Id* candidate_end, rbmd::Id* candidates,
              std::uint64_t candidate_storage_size,
              rbmd::Id* total_candidate_num, rbmd::Id* max_candidate_num,
              Box box, rbmd::Id* error_code);

  void Fill(const rbmd::Id* per_atom_cell_id,
            const rbmd::Id* in_atom_list_start_index,
            const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
            rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
            rbmd::Id total_cells, const rbmd::Real* px,
            const rbmd::Real* py, const rbmd::Real* pz,
            const rbmd::Id* sorted_atom_indices,
            const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
            const rbmd::Real* sorted_pz,
            const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
            const rbmd::Id* candidate_num,
            const rbmd::Id* candidate_start, rbmd::Id* candidate_end,
            rbmd::Id* candidates, std::uint64_t candidate_storage_size,
            Box box, rbmd::Id* error_code);
};

template <>
struct GenerateRblCompactCandidatesOp<device::DEVICE_GPU> {
  void Count(const rbmd::Id* per_atom_cell_id,
             const rbmd::Id* in_atom_list_start_index,
             const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
             rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
             rbmd::Id total_cells, const rbmd::Real* px,
             const rbmd::Real* py, const rbmd::Real* pz,
             const rbmd::Id* sorted_atom_indices,
             const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
             const rbmd::Real* sorted_pz,
             const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
             rbmd::Id* candidate_num, rbmd::Id* max_candidate_num,
             Box box, rbmd::Id* error_code);

  void Append(const rbmd::Id* per_atom_cell_id,
              const rbmd::Id* in_atom_list_start_index,
              const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
              rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
              rbmd::Id total_cells, const rbmd::Real* px,
              const rbmd::Real* py, const rbmd::Real* pz,
              const rbmd::Id* sorted_atom_indices,
              const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
              const rbmd::Real* sorted_pz,
              const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
              rbmd::Id* candidate_num, rbmd::Id* candidate_start,
              rbmd::Id* candidate_end, rbmd::Id* candidates,
              std::uint64_t candidate_storage_size,
              rbmd::Id* total_candidate_num, rbmd::Id* max_candidate_num,
              Box box, rbmd::Id* error_code);

  void Fill(const rbmd::Id* per_atom_cell_id,
            const rbmd::Id* in_atom_list_start_index,
            const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
            rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
            rbmd::Id total_cells, const rbmd::Real* px,
            const rbmd::Real* py, const rbmd::Real* pz,
            const rbmd::Id* sorted_atom_indices,
            const rbmd::Real* sorted_px, const rbmd::Real* sorted_py,
            const rbmd::Real* sorted_pz,
            const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
            const rbmd::Id* candidate_num,
            const rbmd::Id* candidate_start, rbmd::Id* candidate_end,
            rbmd::Id* candidates, std::uint64_t candidate_storage_size,
            Box box, rbmd::Id* error_code);
};

template <typename DEVICE>
struct GenerateRblFixedCandidatesOp {
  void operator()(const rbmd::Id* per_atom_cell_id,
                  const rbmd::Id* in_atom_list_start_index,
                  const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
                  rbmd::Real core_cutoff_2,
                  rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
                  rbmd::Id total_cells, const rbmd::Real* px,
                  const rbmd::Real* py, const rbmd::Real* pz,
                  const rbmd::Id* sorted_atom_indices,
                  const rbmd::Real* sorted_px,
                  const rbmd::Real* sorted_py,
                  const rbmd::Real* sorted_pz,
                  const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
                  rbmd::Id candidate_capacity,
                  const rbmd::Id* candidate_start, rbmd::Id* candidate_end,
                  rbmd::Id* candidates, rbmd::Id* candidate_num,
                  bool implicit_offsets,
                  std::uint64_t candidate_storage_size,
                  rbmd::Id core_capacity, const rbmd::Id* core_start,
                  rbmd::Id* core_end, rbmd::Id* core_neighbors,
                  rbmd::Id* core_neighbor_num,
                  std::uint64_t core_storage_size, Box box,
                  rbmd::Id* status, rbmd::Id threads_per_particle);
};

template <>
struct GenerateRblFixedCandidatesOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Id* per_atom_cell_id,
                  const rbmd::Id* in_atom_list_start_index,
                  const rbmd::Id* in_atom_list_end_index, rbmd::Real cutoff_2,
                  rbmd::Real core_cutoff_2,
                  rbmd::Id active_atom_num, rbmd::Id total_storage_atoms,
                  rbmd::Id total_cells, const rbmd::Real* px,
                  const rbmd::Real* py, const rbmd::Real* pz,
                  const rbmd::Id* sorted_atom_indices,
                  const rbmd::Real* sorted_px,
                  const rbmd::Real* sorted_py,
                  const rbmd::Real* sorted_pz,
                  const rbmd::Id* neighbor_cell, rbmd::Id neighbor_cell_num,
                  rbmd::Id candidate_capacity,
                  const rbmd::Id* candidate_start, rbmd::Id* candidate_end,
                  rbmd::Id* candidates, rbmd::Id* candidate_num,
                  bool implicit_offsets,
                  std::uint64_t candidate_storage_size,
                  rbmd::Id core_capacity, const rbmd::Id* core_start,
                  rbmd::Id* core_end, rbmd::Id* core_neighbors,
                  rbmd::Id* core_neighbor_num,
                  std::uint64_t core_storage_size, Box box,
                  rbmd::Id* status, rbmd::Id threads_per_particle);
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
                  const rbmd::Id* candidate_num,
                  rbmd::Id candidate_capacity, bool implicit_offsets,
                  const rbmd::Id* max_core_neighbor_num,
                  const rbmd::Id* core_start, rbmd::Id* core_end,
                  rbmd::Id* core_neighbors, rbmd::Id* core_neighbor_num,
                  rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors,
                  rbmd::Id* random_neighbors_num,
                  rbmd::Id* required_random_neighbor_capacity,
                  rbmd::Id* required_core_neighbor_capacity, Box box,
                  rbmd::Id* should_realloc, rbmd::Id selection_frequency,
                  rbmd::Id sample_epoch, rbmd::Id threads_per_particle);
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
                  const rbmd::Id* candidate_num,
                  rbmd::Id candidate_capacity, bool implicit_offsets,
                  const rbmd::Id* max_core_neighbor_num,
                  const rbmd::Id* core_start, rbmd::Id* core_end,
                  rbmd::Id* core_neighbors, rbmd::Id* core_neighbor_num,
                  rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors,
                  rbmd::Id* random_neighbors_num,
                  rbmd::Id* required_random_neighbor_capacity,
                  rbmd::Id* required_core_neighbor_capacity, Box box,
                  rbmd::Id* should_realloc, rbmd::Id selection_frequency,
                  rbmd::Id sample_epoch, rbmd::Id threads_per_particle);
};

template <typename DEVICE>
struct SampleRblNeighborCandidatesOp {
  void operator()(rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
                  rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz,
                  const rbmd::Id* candidate_start,
                  const rbmd::Id* candidate_end,
                  const rbmd::Id* candidate_neighbors,
                  const rbmd::Id* candidate_num,
                  rbmd::Id candidate_capacity, bool implicit_offsets,
                  rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors,
                  rbmd::Id* random_neighbors_num,
                  rbmd::Real* random_neighbor_weight,
                  rbmd::Id* required_random_neighbor_capacity,
                  Box box, rbmd::Id selection_frequency,
                  rbmd::Id sample_epoch,
                  rbmd::Id threads_per_particle);
};

template <>
struct SampleRblNeighborCandidatesOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real core_cutoff_2, rbmd::Real force_cutoff_2,
                  rbmd::Id total_atom_num, const rbmd::Id* atoms_id,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz,
                  const rbmd::Id* candidate_start,
                  const rbmd::Id* candidate_end,
                  const rbmd::Id* candidate_neighbors,
                  const rbmd::Id* candidate_num,
                  rbmd::Id candidate_capacity, bool implicit_offsets,
                  rbmd::Id random_neighbor_capacity,
                  rbmd::Id* random_neighbors,
                  rbmd::Id* random_neighbors_num,
                  rbmd::Real* random_neighbor_weight,
                  rbmd::Id* required_random_neighbor_capacity,
                  Box box, rbmd::Id selection_frequency,
                  rbmd::Id sample_epoch,
                  rbmd::Id threads_per_particle);
};

// Independent fixed-p branch. Candidate row ordering changes, core rows do not.
template <typename DEVICE> struct PrepareRblFixedpRowsOp;
template <> struct PrepareRblFixedpRowsOp<device::DEVICE_GPU> {
  void operator()(rbmd::Id owned, rbmd::Id storage_atoms,
      const rbmd::Id* atom_ids, const rbmd::Id* atom_types,
      const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
      const rbmd::Real* pz, const rbmd::Id* candidate_start,
      const rbmd::Id* candidate_end, rbmd::Id* candidates,
      const rbmd::Id* candidate_num, rbmd::Id candidate_capacity,
      bool implicit_offsets, std::uint64_t candidate_storage_size,
      const rbmd::Id* core_start, const rbmd::Id* core_end,
      const rbmd::Id* core_neighbors, std::uint64_t core_storage_size,
      rbmd::Real core_superset_cutoff_2, rbmd::Real physical_core_2, Box box, bool cvff_distance, rbmd::Id* error_code);
};
template <typename DEVICE> struct SampleRblFixedpRowsOp;
template <> struct SampleRblFixedpRowsOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real core_2, rbmd::Real outer_2,
      rbmd::Id owned, rbmd::Id storage_atoms, const rbmd::Id* atom_ids,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      const rbmd::Id* candidate_start, const rbmd::Id* candidate_end,
      const rbmd::Id* candidates, const rbmd::Id* candidate_num,
      rbmd::Id candidate_capacity, bool implicit_offsets,
      rbmd::Id requested, rbmd::Id random_capacity, rbmd::Id* output,
      rbmd::Id* selected_num, rbmd::Id* shell_population,
      rbmd::Real* weight, rbmd::Id* required_capacity, Box box, bool cvff_distance, rbmd::Id epoch, rbmd::Id* error_code);
};


// Failure-only evidence; does not change any neighbor/force predicate.
template <typename DEVICE> struct DiagnoseRblFixedpMissingCoreOp;
template <> struct DiagnoseRblFixedpMissingCoreOp<device::DEVICE_GPU> {
  void operator()(rbmd::Id owned, rbmd::Id storage_atoms,
      const rbmd::Id* atom_ids, const rbmd::Real* px, const rbmd::Real* py,
      const rbmd::Real* pz, const rbmd::Id* candidate_start,
      const rbmd::Id* candidate_end, const rbmd::Id* candidates,
      const rbmd::Id* candidate_num, rbmd::Id candidate_capacity,
      bool implicit_offsets, std::uint64_t candidate_storage_size,
      const rbmd::Id* core_start, const rbmd::Id* core_end,
      const rbmd::Id* core_neighbors, std::uint64_t core_storage_size,
      rbmd::Real core_superset_2, rbmd::Real physical_core_2, Box box, bool cvff_distance,
      unsigned int* claimed, rbmd::Id* witness_ids, rbmd::Real* witness_values);
};

}  // namespace op
