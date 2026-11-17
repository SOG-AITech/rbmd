
#include "neighbor_list_builder/full_neighbor_list_builder.h"

#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <output/include/Logger.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/device_types.h"
#include "common/neighbor_skin.h"
#include "common/types.h"
#include "data_manager.h"
#include "force_box_selector.h"
#include "full_neighbor_list_op.h"
#include "simulate_pipeline/src/controller/position_controller/position_controller_op/update_position_op.h"
#ifdef USE_MPI
#include "rbmd_parallel_until_locator.h"
#endif

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

extern rbmd::Id test_current_step;

namespace {

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

int EnvIntOrDefault(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return fallback;
  }
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

int CurrentProcessId() {
#if defined(_WIN32)
  return static_cast<int>(_getpid());
#else
  return static_cast<int>(::getpid());
#endif
}

void EnsureNeighborDebugLogDirectory() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
#if defined(_WIN32)
  (void)_mkdir("logs");
  (void)_mkdir("logs\\debug");
#else
  (void)mkdir("logs", 0755);
  (void)mkdir("logs/debug", 0755);
#endif
  initialized = true;
}

std::vector<rbmd::Id> ParseTrackedGidListFromEnv(const char* name) {
  std::vector<rbmd::Id> gids;
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return gids;
  }

  std::string normalized(raw);
  for (char& ch : normalized) {
    if (ch == ',' || ch == ';' || ch == '|' || ch == '\t') {
      ch = ' ';
    }
  }

  std::istringstream iss(normalized);
  std::string token;
  while (iss >> token) {
    char* end = nullptr;
    const long long parsed = std::strtoll(token.c_str(), &end, 10);
    if (end == token.c_str() || (end && *end != '\0') || parsed < 0) {
      continue;
    }
    gids.push_back(static_cast<rbmd::Id>(parsed));
  }

  std::sort(gids.begin(), gids.end());
  gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
  return gids;
}

struct NeighborDebugConfig {
  bool enabled{false};
  int every{1};
  std::vector<rbmd::Id> gids{};
};

NeighborDebugConfig ParseNeighborDebugConfig() {
  NeighborDebugConfig cfg;
  cfg.enabled = EnvFlagEnabled("RBMD_DEBUG_NEIGHBOR_STATS");
  cfg.every = EnvIntOrDefault("RBMD_DEBUG_NEIGHBOR_STATS_EVERY", 1);
  cfg.gids = ParseTrackedGidListFromEnv("RBMD_DEBUG_TRACK_GIDS");
  if (cfg.gids.empty()) {
    cfg.gids = ParseTrackedGidListFromEnv("RBMD_DEBUG_EXCHANGE_ROUTE_GIDS");
  }
  return cfg;
}

const NeighborDebugConfig& GetNeighborDebugConfig() {
  static const NeighborDebugConfig cfg = ParseNeighborDebugConfig();
  return cfg;
}

bool ShouldWriteNeighborDebug(rbmd::Id step) {
  const auto& cfg = GetNeighborDebugConfig();
  return cfg.enabled && cfg.every > 0 && (step % cfg.every) == 0;
}

int CurrentRank() {
#ifdef USE_MPI
  if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
    return GET_RBMD_PARALLEL->_domdec->_current_rank;
  }
#endif
  return 0;
}

const char* AtomRegionName(rbmd::Id idx, rbmd::Id nlocal, rbmd::Id ntotal) {
  if (idx < 0) {
    return "missing";
  }
  if (idx < nlocal) {
    return "native";
  }
  if (idx < ntotal) {
    return "ghost";
  }
  return "out_of_total";
}

rbmd::Id ClampNeighborOffset(rbmd::Id raw, std::size_t limit) {
  if (raw < 0) {
    return 0;
  }
  const auto capped = static_cast<std::size_t>(raw);
  return static_cast<rbmd::Id>(capped > limit ? limit : capped);
}

rbmd::Id EffectiveNeighborCellNum(const LinkedCell& linked_cell) {
  const rbmd::Id stencil_width =
      2 * linked_cell._cell_count_within_cutoff + 1;
  rbmd::Id result = 1;
  for (int dim = 0; dim < 3; ++dim) {
    const rbmd::Id available = linked_cell._covers_whole_domain[dim]
                                   ? linked_cell._nativate_cells[dim]
                                   : linked_cell._per_dimension_cells[dim];
    result *= std::min(stencil_width, available);
  }
  return result;
}

bool CanUseCompactNeighborStencil(const LinkedCell& linked_cell) {
  const rbmd::Id radius = linked_cell._cell_count_within_cutoff;
  const rbmd::Id stencil_width = 2 * radius + 1;
  if (radius <= 0 || !(linked_cell._neighbor_cutoff > rbmd::Real(0))) {
    return false;
  }
  for (int dim = 0; dim < 3; ++dim) {
    const rbmd::Id available = linked_cell._covers_whole_domain[dim]
                                   ? linked_cell._nativate_cells[dim]
                                   : linked_cell._per_dimension_cells[dim];
    if (available < stencil_width ||
        !(linked_cell._cell_length[dim] > rbmd::Real(0))) {
      return false;
    }
  }
  return true;
}

std::vector<rbmd::Id> BuildNeighborStencilIndices(
    const LinkedCell& linked_cell) {
  const rbmd::Id full_stencil_size = EffectiveNeighborCellNum(linked_cell);
  if (full_stencil_size <= 0) {
    return {};
  }

  std::vector<rbmd::Id> stencil_indices;
  stencil_indices.reserve(static_cast<std::size_t>(full_stencil_size));
  if (!CanUseCompactNeighborStencil(linked_cell)) {
    for (rbmd::Id idx = 0; idx < full_stencil_size; ++idx) {
      stencil_indices.push_back(idx);
    }
    return stencil_indices;
  }

  const rbmd::Id radius = linked_cell._cell_count_within_cutoff;
  const rbmd::Id stencil_width = 2 * radius + 1;
  const rbmd::Real cutoff_sq =
      linked_cell._neighbor_cutoff * linked_cell._neighbor_cutoff;
  const auto axis_gap = [](rbmd::Id offset) {
    return std::max<rbmd::Id>(std::abs(offset) - rbmd::Id(1), rbmd::Id(0));
  };
  for (rbmd::Id raw_idx = 0; raw_idx < full_stencil_size; ++raw_idx) {
    const rbmd::Id offset_x = raw_idx % stencil_width - radius;
    const rbmd::Id offset_y =
        (raw_idx / stencil_width) % stencil_width - radius;
    const rbmd::Id offset_z =
        raw_idx / (stencil_width * stencil_width) - radius;
    const rbmd::Real gap_x = static_cast<rbmd::Real>(axis_gap(offset_x)) *
                             linked_cell._cell_length[0];
    const rbmd::Real gap_y = static_cast<rbmd::Real>(axis_gap(offset_y)) *
                             linked_cell._cell_length[1];
    const rbmd::Real gap_z = static_cast<rbmd::Real>(axis_gap(offset_z)) *
                             linked_cell._cell_length[2];
    const rbmd::Real minimum_distance_sq =
        gap_x * gap_x + gap_y * gap_y + gap_z * gap_z;
    if (minimum_distance_sq < cutoff_sq) {
      stencil_indices.push_back(raw_idx);
    }
  }
  return stencil_indices;
}

void EmitNeighborDebugCsv(const LinkedCell& linked_cell,
                          const NeighborList& neighbor_list,
                          const std::shared_ptr<DeviceData>& device_data,
                          rbmd::Id neighbor_cell_num) {
  if (!device_data || !ShouldWriteNeighborDebug(test_current_step)) {
    return;
  }

  const rbmd::Id nlocal = std::max<rbmd::Id>(linked_cell._native_atoms_num, 0);
  const rbmd::Id nghost = std::max<rbmd::Id>(linked_cell._ghost_atoms_num, 0);
  const rbmd::Id ntotal = std::max<rbmd::Id>(linked_cell._total_atoms_num, 0);
  const std::size_t native_size = static_cast<std::size_t>(nlocal);
  const std::size_t total_size = static_cast<std::size_t>(ntotal);
  const std::size_t ids_size =
      std::min(total_size, device_data->_d_atoms_id.size());
  const std::size_t cell_id_size =
      std::min(total_size, linked_cell._per_atom_cell_id.size());
  const std::size_t cell_sorted_size =
      std::min(total_size, linked_cell._cell_sorted_cell_id.size());
  const std::size_t start_size =
      std::min(native_size, neighbor_list._start_idx.size());
  const std::size_t end_size =
      std::min(native_size, neighbor_list._end_idx.size());
  const std::size_t estimated_size =
      std::min(native_size, neighbor_list._d_neighbor_num.size());
  const std::size_t max_neighbor_size =
      std::min(native_size, neighbor_list._d_max_neighbor_num.size());
  const std::size_t neighbor_cell_size = linked_cell._neighbor_cell.size();
  const std::size_t sorted_atom_index_size =
      std::min(total_size, linked_cell._cell_sorted_atom_indices.size());

  thrust::host_vector<rbmd::Id> h_ids(
      device_data->_d_atoms_id.begin(),
      device_data->_d_atoms_id.begin() + ids_size);
  thrust::host_vector<rbmd::Id> h_cell_ids(
      linked_cell._per_atom_cell_id.begin(),
      linked_cell._per_atom_cell_id.begin() + cell_id_size);
  thrust::host_vector<rbmd::Id> h_cell_sorted_ids(
      linked_cell._cell_sorted_cell_id.begin(),
      linked_cell._cell_sorted_cell_id.begin() + cell_sorted_size);
  thrust::host_vector<Cell> h_cells = linked_cell._cells;
  thrust::host_vector<rbmd::Id> h_cell_start =
      linked_cell._in_atom_list_start_index;
  thrust::host_vector<rbmd::Id> h_cell_end =
      linked_cell._in_atom_list_end_index;
  thrust::host_vector<rbmd::Real> h_px(device_data->_d_px.begin(),
                                       device_data->_d_px.begin() + ids_size);
  thrust::host_vector<rbmd::Real> h_py(device_data->_d_py.begin(),
                                       device_data->_d_py.begin() + ids_size);
  thrust::host_vector<rbmd::Real> h_pz(device_data->_d_pz.begin(),
                                       device_data->_d_pz.begin() + ids_size);
  thrust::host_vector<rbmd::Id> h_start(
      neighbor_list._start_idx.begin(),
      neighbor_list._start_idx.begin() + start_size);
  thrust::host_vector<rbmd::Id> h_end(
      neighbor_list._end_idx.begin(),
      neighbor_list._end_idx.begin() + end_size);
  thrust::host_vector<rbmd::Id> h_estimated(
      neighbor_list._d_neighbor_num.begin(),
      neighbor_list._d_neighbor_num.begin() + estimated_size);
  thrust::host_vector<rbmd::Id> h_max_neighbor(
      neighbor_list._d_max_neighbor_num.begin(),
      neighbor_list._d_max_neighbor_num.begin() + max_neighbor_size);
  thrust::host_vector<rbmd::Id> h_neighbors = neighbor_list._d_neighbors;
  thrust::host_vector<rbmd::Id> h_neighbor_cells = linked_cell._neighbor_cell;
  thrust::host_vector<rbmd::Id> h_sorted_atom_indices(
      linked_cell._cell_sorted_atom_indices.begin(),
      linked_cell._cell_sorted_atom_indices.begin() + sorted_atom_index_size);

  rbmd::Id per_atom_cell_descents = 0;
  rbmd::Id first_per_atom_descent_idx = -1;
  rbmd::Id first_per_atom_descent_prev = -1;
  rbmd::Id first_per_atom_descent_curr = -1;
  for (std::size_t idx = 1; idx < h_cell_ids.size(); ++idx) {
    if (h_cell_ids[idx] < h_cell_ids[idx - 1]) {
      ++per_atom_cell_descents;
      if (first_per_atom_descent_idx < 0) {
        first_per_atom_descent_idx = static_cast<rbmd::Id>(idx);
        first_per_atom_descent_prev = h_cell_ids[idx - 1];
        first_per_atom_descent_curr = h_cell_ids[idx];
      }
    }
  }

  rbmd::Id cell_sorted_descents = 0;
  rbmd::Id first_cell_sorted_descent_idx = -1;
  rbmd::Id first_cell_sorted_descent_prev = -1;
  rbmd::Id first_cell_sorted_descent_curr = -1;
  for (std::size_t idx = 1; idx < h_cell_sorted_ids.size(); ++idx) {
    if (h_cell_sorted_ids[idx] < h_cell_sorted_ids[idx - 1]) {
      ++cell_sorted_descents;
      if (first_cell_sorted_descent_idx < 0) {
        first_cell_sorted_descent_idx = static_cast<rbmd::Id>(idx);
        first_cell_sorted_descent_prev = h_cell_sorted_ids[idx - 1];
        first_cell_sorted_descent_curr = h_cell_sorted_ids[idx];
      }
    }
  }

  std::vector<rbmd::Id> actual_cell_counts(h_cells.size(), 0);
  for (const rbmd::Id cell_id : h_cell_sorted_ids) {
    if (cell_id >= 0 &&
        static_cast<std::size_t>(cell_id) < actual_cell_counts.size()) {
      ++actual_cell_counts[static_cast<std::size_t>(cell_id)];
    }
  }

  rbmd::Id range_mismatch_cells = 0;
  rbmd::Id first_range_mismatch_cell = -1;
  rbmd::Id first_range_mismatch_range_count = -1;
  rbmd::Id first_range_mismatch_actual_count = -1;
  rbmd::Id total_range_atoms = 0;
  rbmd::Id total_actual_cell_atoms = 0;
  for (std::size_t cell = 0; cell < h_cells.size(); ++cell) {
    const rbmd::Id begin =
        ClampNeighborOffset(h_cell_start[cell], h_cell_sorted_ids.size());
    const rbmd::Id end =
        ClampNeighborOffset(h_cell_end[cell], h_cell_sorted_ids.size());
    const rbmd::Id range_count = std::max<rbmd::Id>(0, end - begin);
    const rbmd::Id actual_count = actual_cell_counts[cell];
    total_range_atoms += range_count;
    total_actual_cell_atoms += actual_count;
    if (range_count != actual_count) {
      ++range_mismatch_cells;
      if (first_range_mismatch_cell < 0) {
        first_range_mismatch_cell = static_cast<rbmd::Id>(cell);
        first_range_mismatch_range_count = range_count;
        first_range_mismatch_actual_count = actual_count;
      }
    }
  }

  long long total_neighbor_refs = 0;
  long long ghost_neighbor_refs = 0;
  long long self_neighbor_refs = 0;
  rbmd::Id max_neighbor_refs = 0;
  rbmd::Id first_self_atom_idx = -1;
  rbmd::Id first_self_atom_gid = -1;
  rbmd::Id first_self_neighbor_slot = -1;
  for (std::size_t atom_idx = 0;
       atom_idx < native_size && atom_idx < start_size && atom_idx < end_size;
       ++atom_idx) {
    const rbmd::Id safe_start =
        ClampNeighborOffset(h_start[atom_idx], h_neighbors.size());
    const rbmd::Id safe_end =
        ClampNeighborOffset(h_end[atom_idx], h_neighbors.size());
    const rbmd::Id begin = std::min(safe_start, safe_end);
    const rbmd::Id end = std::max(safe_start, safe_end);
    const rbmd::Id neighbor_count = end - begin;
    total_neighbor_refs += static_cast<long long>(neighbor_count);
    if (neighbor_count > max_neighbor_refs) {
      max_neighbor_refs = neighbor_count;
    }
    for (rbmd::Id slot = begin; slot < end; ++slot) {
      const rbmd::Id neighbor_idx = h_neighbors[static_cast<std::size_t>(slot)];
      if (neighbor_idx == static_cast<rbmd::Id>(atom_idx)) {
        ++self_neighbor_refs;
        if (first_self_atom_idx < 0) {
          first_self_atom_idx = static_cast<rbmd::Id>(atom_idx);
          first_self_atom_gid =
              (atom_idx < h_ids.size()) ? h_ids[atom_idx] : rbmd::Id(-1);
          first_self_neighbor_slot = slot;
        }
      }
      if (neighbor_idx >= nlocal && neighbor_idx < ntotal) {
        ++ghost_neighbor_refs;
      }
    }
  }

  const double mean_neighbor_refs =
      nlocal > 0 ? static_cast<double>(total_neighbor_refs) /
                       static_cast<double>(nlocal)
                 : 0.0;

  EnsureNeighborDebugLogDirectory();
  const int rank = CurrentRank();
  const int pid = CurrentProcessId();
  const Box force_box = GetPeriodicBoxForNeighborAndForce(
      *(DataManager::getInstance().getMDData()->_box));
  const rbmd::Real cutoff_sq =
      linked_cell._neighbor_cutoff * linked_cell._neighbor_cutoff;

  {
    const std::string filename = "logs/debug/neighbor_stats_rank" +
                                 std::to_string(rank) + "_pid" +
                                 std::to_string(pid) + ".csv";
    std::ofstream csv(filename, std::ios::app);
    if (csv.is_open()) {
      if (csv.tellp() == 0) {
        csv << "step,rank,nlocal,nghost,ntotal,total_neighbor_refs,"
               "ghost_neighbor_refs,self_neighbor_refs,first_self_atom_idx,"
               "first_self_atom_gid,first_self_neighbor_slot,"
               "mean_neighbor_refs,max_neighbor_refs,"
               "per_atom_cell_descents,first_per_atom_descent_idx,"
               "first_per_atom_descent_prev,first_per_atom_descent_curr,"
               "cell_sorted_descents,first_cell_sorted_descent_idx,"
               "first_cell_sorted_descent_prev,first_cell_sorted_descent_curr,"
               "range_mismatch_cells,first_range_mismatch_cell,"
               "first_range_mismatch_range_count,"
               "first_range_mismatch_actual_count,total_range_atoms,"
               "total_actual_cell_atoms\n";
      }
      csv << test_current_step << "," << rank << "," << nlocal << "," << nghost
          << "," << ntotal << "," << total_neighbor_refs << ","
          << ghost_neighbor_refs << "," << self_neighbor_refs << ","
          << first_self_atom_idx << "," << first_self_atom_gid << ","
          << first_self_neighbor_slot << "," << mean_neighbor_refs << ","
          << max_neighbor_refs << "," << per_atom_cell_descents << ","
          << first_per_atom_descent_idx << "," << first_per_atom_descent_prev
          << "," << first_per_atom_descent_curr << "," << cell_sorted_descents
          << "," << first_cell_sorted_descent_idx << ","
          << first_cell_sorted_descent_prev << ","
          << first_cell_sorted_descent_curr << "," << range_mismatch_cells
          << "," << first_range_mismatch_cell << ","
          << first_range_mismatch_range_count << ","
          << first_range_mismatch_actual_count << "," << total_range_atoms
          << "," << total_actual_cell_atoms << "\n";
    }
  }

  const auto& cfg = GetNeighborDebugConfig();
  if (cfg.gids.empty()) {
    return;
  }

  std::unordered_map<rbmd::Id, std::size_t> gid_to_slot;
  gid_to_slot.reserve(cfg.gids.size());
  for (std::size_t i = 0; i < cfg.gids.size(); ++i) {
    gid_to_slot.emplace(cfg.gids[i], i);
  }

  std::vector<int> local_counts(cfg.gids.size(), 0);
  std::vector<rbmd::Id> first_indices(cfg.gids.size(), rbmd::Id(-1));
  for (std::size_t idx = 0; idx < ids_size; ++idx) {
    const auto it = gid_to_slot.find(h_ids[idx]);
    if (it == gid_to_slot.end()) {
      continue;
    }
    const std::size_t slot = it->second;
    local_counts[slot] += 1;
    if (first_indices[slot] < 0) {
      first_indices[slot] = static_cast<rbmd::Id>(idx);
    }
  }

  const std::string filename = "logs/debug/neighbor_gid_stats_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,nlocal,nghost,ntotal,gid,local_count,first_idx,"
           "first_region,cell_idx,cell_is_halo,neighbor_count,"
           "ghost_neighbor_count,self_neighbor_count,first_self_slot,"
           "estimated_neighbor_num,max_neighbor_num,neighbor_start,neighbor_"
           "end\n";
  }

  const std::string stencil_filename =
      "logs/debug/neighbor_gid_stencil_stats_rank" + std::to_string(rank) +
      "_pid" + std::to_string(pid) + ".csv";
  std::ofstream stencil_csv(stencil_filename, std::ios::app);
  if (stencil_csv.is_open() && stencil_csv.tellp() == 0) {
    stencil_csv
        << "step,rank,nlocal,nghost,ntotal,gid,first_idx,first_region,cell_idx,"
           "cell_is_halo,stencil_cell_count,stencil_nonempty_cells,"
           "stencil_range_atom_count,stencil_native_atom_count,"
           "stencil_ghost_atom_count,stencil_self_atom_count,"
           "stencil_min_distance_sq,stencil_cutoff_candidate_count,"
           "stencil_cutoff_ghost_candidate_count,"
           "first_nonempty_neighbor_cell,first_nonempty_range_count\n";
  }

  for (std::size_t i = 0; i < cfg.gids.size(); ++i) {
    const rbmd::Id first_idx = first_indices[i];
    rbmd::Id cell_idx = -1;
    int cell_is_halo = -1;
    rbmd::Id neighbor_count = -1;
    rbmd::Id ghost_neighbor_count = -1;
    rbmd::Id self_neighbor_count = -1;
    rbmd::Id first_self_slot = -1;
    rbmd::Id estimated_neighbor_num = -1;
    rbmd::Id max_neighbor_num = -1;
    rbmd::Id neighbor_start_idx = -1;
    rbmd::Id neighbor_end_idx = -1;
    if (first_idx >= 0 && static_cast<std::size_t>(first_idx) < cell_id_size) {
      cell_idx = h_cell_ids[static_cast<std::size_t>(first_idx)];
      if (cell_idx >= 0 &&
          static_cast<std::size_t>(cell_idx) < h_cells.size()) {
        cell_is_halo =
            h_cells[static_cast<std::size_t>(cell_idx)]._is_halo ? 1 : 0;
      }
    }
    if (first_idx >= 0 && first_idx < nlocal &&
        static_cast<std::size_t>(first_idx) < start_size &&
        static_cast<std::size_t>(first_idx) < end_size) {
      if (static_cast<std::size_t>(first_idx) < estimated_size) {
        estimated_neighbor_num =
            h_estimated[static_cast<std::size_t>(first_idx)];
      }
      if (static_cast<std::size_t>(first_idx) < max_neighbor_size) {
        max_neighbor_num = h_max_neighbor[static_cast<std::size_t>(first_idx)];
      }
      const rbmd::Id safe_start = ClampNeighborOffset(
          h_start[static_cast<std::size_t>(first_idx)], h_neighbors.size());
      const rbmd::Id safe_end = ClampNeighborOffset(
          h_end[static_cast<std::size_t>(first_idx)], h_neighbors.size());
      const rbmd::Id begin = std::min(safe_start, safe_end);
      const rbmd::Id end = std::max(safe_start, safe_end);
      neighbor_start_idx = begin;
      neighbor_end_idx = end;
      neighbor_count = end - begin;
      ghost_neighbor_count = 0;
      self_neighbor_count = 0;
      for (rbmd::Id slot = begin; slot < end; ++slot) {
        const rbmd::Id neighbor_idx =
            h_neighbors[static_cast<std::size_t>(slot)];
        if (neighbor_idx == first_idx) {
          ++self_neighbor_count;
          if (first_self_slot < 0) {
            first_self_slot = slot;
          }
        }
        if (neighbor_idx >= nlocal && neighbor_idx < ntotal) {
          ++ghost_neighbor_count;
        }
      }
    }

    csv << test_current_step << "," << rank << "," << nlocal << "," << nghost
        << "," << ntotal << "," << cfg.gids[i] << "," << local_counts[i] << ","
        << first_idx << "," << AtomRegionName(first_idx, nlocal, ntotal) << ","
        << cell_idx << "," << cell_is_halo << "," << neighbor_count << ","
        << ghost_neighbor_count << "," << self_neighbor_count << ","
        << first_self_slot << "," << estimated_neighbor_num << ","
        << max_neighbor_num << "," << neighbor_start_idx << ","
        << neighbor_end_idx << "\n";

    if (!stencil_csv.is_open()) {
      continue;
    }

    rbmd::Id stencil_cell_count = -1;
    rbmd::Id stencil_nonempty_cells = -1;
    rbmd::Id stencil_range_atom_count = -1;
    rbmd::Id stencil_native_atom_count = -1;
    rbmd::Id stencil_ghost_atom_count = -1;
    rbmd::Id stencil_self_atom_count = -1;
    rbmd::Real stencil_min_distance_sq = rbmd::Real(-1);
    rbmd::Id stencil_cutoff_candidate_count = -1;
    rbmd::Id stencil_cutoff_ghost_candidate_count = -1;
    rbmd::Id first_nonempty_neighbor_cell = -1;
    rbmd::Id first_nonempty_range_count = -1;
    if (cell_idx >= 0 && static_cast<std::size_t>(cell_idx) < h_cells.size()) {
      stencil_cell_count = neighbor_cell_num;
      stencil_nonempty_cells = 0;
      stencil_range_atom_count = 0;
      stencil_native_atom_count = 0;
      stencil_ghost_atom_count = 0;
      stencil_self_atom_count = 0;
      stencil_cutoff_candidate_count = 0;
      stencil_cutoff_ghost_candidate_count = 0;
      rbmd::Real min_distance_sq = std::numeric_limits<rbmd::Real>::max();
      const std::size_t cell_offset =
          static_cast<std::size_t>(cell_idx) *
          static_cast<std::size_t>(stencil_cell_count);
      for (rbmd::Id offset = 0; offset < stencil_cell_count; ++offset) {
        const std::size_t neighbour_cell_slot =
            cell_offset + static_cast<std::size_t>(offset);
        if (neighbour_cell_slot >= neighbor_cell_size) {
          break;
        }
        const rbmd::Id neighbor_cell_id = h_neighbor_cells[neighbour_cell_slot];
        if (neighbor_cell_id < 0 ||
            static_cast<std::size_t>(neighbor_cell_id) >= h_cells.size()) {
          continue;
        }
        const rbmd::Id begin = ClampNeighborOffset(
            h_cell_start[static_cast<std::size_t>(neighbor_cell_id)],
            h_sorted_atom_indices.size());
        const rbmd::Id end = ClampNeighborOffset(
            h_cell_end[static_cast<std::size_t>(neighbor_cell_id)],
            h_sorted_atom_indices.size());
        const rbmd::Id safe_begin = std::min(begin, end);
        const rbmd::Id safe_end = std::max(begin, end);
        const rbmd::Id range_count = safe_end - safe_begin;
        if (range_count <= 0) {
          continue;
        }
        ++stencil_nonempty_cells;
        stencil_range_atom_count += range_count;
        if (first_nonempty_neighbor_cell < 0) {
          first_nonempty_neighbor_cell = neighbor_cell_id;
          first_nonempty_range_count = range_count;
        }
        for (rbmd::Id pos = safe_begin; pos < safe_end; ++pos) {
          const rbmd::Id atom_idx =
              h_sorted_atom_indices[static_cast<std::size_t>(pos)];
          if (atom_idx == first_idx) {
            ++stencil_self_atom_count;
          }
          if (atom_idx >= 0 && atom_idx < nlocal) {
            ++stencil_native_atom_count;
          } else if (atom_idx >= nlocal && atom_idx < ntotal) {
            ++stencil_ghost_atom_count;
          }
          if (first_idx >= 0 &&
              static_cast<std::size_t>(first_idx) < h_px.size() &&
              atom_idx >= 0 &&
              static_cast<std::size_t>(atom_idx) < h_px.size() &&
              atom_idx != first_idx) {
            rbmd::Real dx = h_px[static_cast<std::size_t>(atom_idx)] -
                            h_px[static_cast<std::size_t>(first_idx)];
            rbmd::Real dy = h_py[static_cast<std::size_t>(atom_idx)] -
                            h_py[static_cast<std::size_t>(first_idx)];
            rbmd::Real dz = h_pz[static_cast<std::size_t>(atom_idx)] -
                            h_pz[static_cast<std::size_t>(first_idx)];
            MinImageDistance(force_box, dx, dy, dz);
            const rbmd::Real distance_sq = dx * dx + dy * dy + dz * dz;
            if (distance_sq < min_distance_sq) {
              min_distance_sq = distance_sq;
            }
            if (distance_sq < cutoff_sq) {
              ++stencil_cutoff_candidate_count;
              if (atom_idx >= nlocal && atom_idx < ntotal) {
                ++stencil_cutoff_ghost_candidate_count;
              }
            }
          }
        }
      }
      if (min_distance_sq != std::numeric_limits<rbmd::Real>::max()) {
        stencil_min_distance_sq = min_distance_sq;
      }
    }

    stencil_csv << test_current_step << "," << rank << "," << nlocal << ","
                << nghost << "," << ntotal << "," << cfg.gids[i] << ","
                << first_idx << "," << AtomRegionName(first_idx, nlocal, ntotal)
                << "," << cell_idx << "," << cell_is_halo << ","
                << stencil_cell_count << "," << stencil_nonempty_cells << ","
                << stencil_range_atom_count << "," << stencil_native_atom_count
                << "," << stencil_ghost_atom_count << ","
                << stencil_self_atom_count << "," << stencil_min_distance_sq
                << "," << stencil_cutoff_candidate_count << ","
                << stencil_cutoff_ghost_candidate_count << ","
                << first_nonempty_neighbor_cell << ","
                << first_nonempty_range_count << "\n";
  }
}

}  // namespace

FullNeighborListBuilder::FullNeighborListBuilder() {
  this->_neighbor_cell_num = EffectiveNeighborCellNum(*_linked_cell);
  this->_neighbor_list =
      std::make_shared<NeighborList>(_linked_cell->_native_atoms_num, false);
  this->_device_data = DataManager::getInstance().getDeviceData();
  this->_d_required_neighbor_capacity.resize(1);
  _trunc_distance_power_2 =
      //_linked_cell->_cutoff * _linked_cell->_cutoff - EPSILON;
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff;
  EnsureNeighborCellsCurrent();
}

std::shared_ptr<NeighborList> FullNeighborListBuilder::Build() {
  const rbmd::Real neighbor_cutoff = _linked_cell->_neighbor_cutoff;
  this->_trunc_distance_power_2 = neighbor_cutoff * neighbor_cutoff;
  if (ShouldReuseNeighborList(neighbor_cutoff)) {
#ifdef USE_MPI
    GET_RBMD_PARALLEL->ForwardExchangeCoordinates();
#endif
    return _neighbor_list;
  }

#ifdef USE_MPI
  _linked_cell->ClearDataHalo();
  RbmdParallelUntilLocator::GetInstance()
      .GetRbmdParallelUntil()
      ->BalanceAndExchange();
  if (this->_neighbor_list->_d_max_neighbor_num.size() <
      static_cast<std::size_t>(_linked_cell->_native_atoms_num)) {
    this->_neighbor_list->resize(_linked_cell->_native_atoms_num);
    this->should_realloc = true;
  }
#endif
  _linked_cell->AssignAtomsToCell();
  _linked_cell->SortAtomsByCellKey();
  _linked_cell->ComputeCellRangesIndices();
  EnsureNeighborCellsCurrent();
  if (should_realloc) {
    EstimateNeighborsList();
  }
  while (GenerateNeighborsList() == RBMD_TRUE) {
    EstimateNeighborsList();
  }
  RecordNeighborBuildState(neighbor_cutoff);
#ifdef USE_MPI
  // Distributed SHAKE needs the current halo's forward map even when
  // skin == 0 disables neighbour-list reuse.
  if (!GET_RBMD_PARALLEL->PrepareForwardCoordinateExchange()) {
    if (HasNeighborCache()) {
      InvalidateNeighborCache();
    }
  }
#endif
  EmitNeighborDebugCsv(*_linked_cell, *this->_neighbor_list, _device_data,
                       _neighbor_cell_num);
  return _neighbor_list;
}

std::shared_ptr<NeighborList> FullNeighborListBuilder::Build(
    rbmd::Real custom_cutoff) {
  const rbmd::Real neighbor_cutoff = rbmd::neighbor::NeighborCutoff(
      custom_cutoff, _linked_cell->_skin, "FullNeighborListBuilder::Build");
  this->_trunc_distance_power_2 = neighbor_cutoff * neighbor_cutoff;
  if (ShouldReuseNeighborList(neighbor_cutoff)) {
#ifdef USE_MPI
    GET_RBMD_PARALLEL->ForwardExchangeCoordinates();
#endif
    return _neighbor_list;
  }

#ifdef USE_MPI
  _linked_cell->ClearDataHalo();
  RbmdParallelUntilLocator::GetInstance()
      .GetRbmdParallelUntil()
      ->BalanceAndExchange();
  if (this->_neighbor_list->_d_max_neighbor_num.size() <
      static_cast<std::size_t>(_linked_cell->_native_atoms_num)) {
    this->_neighbor_list->resize(_linked_cell->_native_atoms_num);
    this->should_realloc = true;
  }
#endif
  _linked_cell->AssignAtomsToCell();
  _linked_cell->SortAtomsByCellKey();
  _linked_cell->ComputeCellRangesIndices();
  EnsureNeighborCellsCurrent();
  if (should_realloc) {
    EstimateNeighborsList();
  }
  while (GenerateNeighborsList() == RBMD_TRUE) {
    EstimateNeighborsList();
  }
  RecordNeighborBuildState(neighbor_cutoff);
#ifdef USE_MPI
  // Distributed SHAKE needs the current halo's forward map even when
  // skin == 0 disables neighbour-list reuse.
  if (!GET_RBMD_PARALLEL->PrepareForwardCoordinateExchange()) {
    if (HasNeighborCache()) {
      InvalidateNeighborCache();
    }
  }
#endif
  EmitNeighborDebugCsv(*_linked_cell, *this->_neighbor_list, _device_data,
                       _neighbor_cell_num);
  return _neighbor_list;
}

rbmd::Id FullNeighborListBuilder::ActiveAtomCount() const {
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  return std::max<rbmd::Id>(active_atoms, 0);
}

rbmd::Id FullNeighborListBuilder::InitialNeighborCapacity(
    rbmd::Real cutoff) const {
  const rbmd::Real cell_volume = _linked_cell->_cell_length[0] *
                                 _linked_cell->_cell_length[1] *
                                 _linked_cell->_cell_length[2];
  const rbmd::Real grid_volume =
      cell_volume * static_cast<rbmd::Real>(_linked_cell->_total_cells);
  rbmd::Real density = 0;
  if (grid_volume > rbmd::Real(0)) {
    density = static_cast<rbmd::Real>(_linked_cell->_total_atoms_num) /
              grid_volume;
  }
  const rbmd::Real expected =
      density * rbmd::Real(4.0 / 3.0) * RBMD_PI * cutoff * cutoff * cutoff;
  rbmd::Id capacity = static_cast<rbmd::Id>(
      std::ceil(static_cast<double>(expected * SAFE_ZONE)));
  capacity = std::max<rbmd::Id>(capacity, MIN_NBNUM);
  return (capacity + 3) & ~rbmd::Id(3);
}

rbmd::Id FullNeighborListBuilder::GrownNeighborCapacity(
    rbmd::Id required) const {
  const rbmd::Id grown = static_cast<rbmd::Id>(
      std::ceil(static_cast<double>(required) * SAFE_ZONE));
  return (std::max<rbmd::Id>(grown, MIN_NBNUM) + 3) & ~rbmd::Id(3);
}

rbmd::Id FullNeighborListBuilder::ChooseThreadsPerParticle() const {
  const long long native_cells =
      static_cast<long long>(_linked_cell->_nativate_cells[0]) *
      static_cast<long long>(_linked_cell->_nativate_cells[1]) *
      static_cast<long long>(_linked_cell->_nativate_cells[2]);
  const rbmd::Id average_cell_occupancy = static_cast<rbmd::Id>(
      _linked_cell->_total_atoms_num / std::max<long long>(native_cells, 1));
  const long long estimated_work =
      static_cast<long long>(average_cell_occupancy) *
      static_cast<long long>(std::max<rbmd::Id>(_neighbor_cell_num, 1));
  return ChooseThreadsPerParticle(
      static_cast<rbmd::Id>(std::min<long long>(
          estimated_work, std::numeric_limits<rbmd::Id>::max())),
      "RBMD_VL_BUILD_TPP", rbmd::Id(1));
}

rbmd::Id FullNeighborListBuilder::ChooseThreadsPerParticle(
    rbmd::Id estimated_work_items, const char* override_env,
    rbmd::Id minimum_tpp) const {
  const char* value =
      override_env == nullptr ? nullptr : std::getenv(override_env);
  if (value != nullptr && value[0] != '\0') {
    const rbmd::Id requested = static_cast<rbmd::Id>(std::atoi(value));
    switch (requested) {
      case 1:
      case 4:
      case 8:
      case 16:
      case 32:
#if WARP_SIZE_64
      case WARP_SIZE:
#endif
        return requested;
      default:
        break;
    }
  }

  // One logical group owns an atom. Neighbor construction is dominated by
  // irregular memory traffic, so retain several dozen checks per lane before
  // widening the group. This preserves latency-hiding across atoms while still
  // widening exceptionally dense rows.
  rbmd::Id threads_per_particle =
      std::max<rbmd::Id>(minimum_tpp, rbmd::Id(1));
  const rbmd::Id required_lanes =
      std::max<rbmd::Id>(
          (std::max<rbmd::Id>(estimated_work_items, 1) + 95) / 96,
          rbmd::Id(1));
  while (threads_per_particle < required_lanes &&
         threads_per_particle < WARP_SIZE) {
    threads_per_particle *= 2;
  }
  return std::min<rbmd::Id>(threads_per_particle, WARP_SIZE);
}

void FullNeighborListBuilder::PrepareFixedWidthList(
    const std::shared_ptr<NeighborList>& list, rbmd::Id active_atoms,
    rbmd::Id capacity) {
  if (active_atoms < 0 || capacity <= 0) {
    throw std::runtime_error("Invalid fixed-width neighbor-list dimensions");
  }
  if (list->_start_idx.size() < static_cast<std::size_t>(active_atoms)) {
    list->resize(active_atoms);
  }

  const unsigned long long storage_size =
      static_cast<unsigned long long>(active_atoms) *
      static_cast<unsigned long long>(capacity);
  if (storage_size >
      static_cast<unsigned long long>(std::numeric_limits<rbmd::Id>::max())) {
    throw std::overflow_error(
        "Fixed-width neighbor-list index overflow: active_atoms=" +
        std::to_string(active_atoms) + ", capacity=" +
        std::to_string(capacity) + ", entries=" +
        std::to_string(storage_size));
  }

  list->_max_neighbor_capacity = capacity;
  list->_h_total_max_neighbor_num = static_cast<rbmd::Id>(storage_size);
  list->_d_neighbors.resize(static_cast<std::size_t>(storage_size));
  thrust::fill(list->_d_max_neighbor_num.begin(),
               list->_d_max_neighbor_num.begin() + active_atoms, capacity);
  thrust::fill(list->_d_neighbor_num.begin(),
               list->_d_neighbor_num.begin() + active_atoms, rbmd::Id(0));
  thrust::sequence(list->_start_idx.begin(),
                   list->_start_idx.begin() + active_atoms, rbmd::Id(0),
                   capacity);
  thrust::copy(list->_start_idx.begin(),
               list->_start_idx.begin() + active_atoms,
               list->_end_idx.begin());
}

std::array<rbmd::Id, 14>
FullNeighborListBuilder::NeighborCellGridSignature() const {
  return {_linked_cell->_total_cells,
          _linked_cell->_cell_count_within_cutoff,
          _linked_cell->_per_dimension_cells[0],
          _linked_cell->_per_dimension_cells[1],
          _linked_cell->_per_dimension_cells[2],
          _linked_cell->_nativate_cells[0],
          _linked_cell->_nativate_cells[1],
          _linked_cell->_nativate_cells[2],
          _linked_cell->_nativate_cell_start[0],
          _linked_cell->_nativate_cell_start[1],
          _linked_cell->_nativate_cell_start[2],
          static_cast<rbmd::Id>(_linked_cell->_covers_whole_domain[0]),
          static_cast<rbmd::Id>(_linked_cell->_covers_whole_domain[1]),
          static_cast<rbmd::Id>(_linked_cell->_covers_whole_domain[2])};
}

std::array<rbmd::Real, 4>
FullNeighborListBuilder::NeighborStencilGeometrySignature() const {
  return {_linked_cell->_cell_length[0], _linked_cell->_cell_length[1],
          _linked_cell->_cell_length[2], _linked_cell->_neighbor_cutoff};
}

void FullNeighborListBuilder::EnsureNeighborCellsCurrent() {
  const auto signature = NeighborCellGridSignature();
  const auto geometry_signature = NeighborStencilGeometrySignature();
  if (_neighbor_cells_valid && signature == _neighbor_cell_grid_signature &&
      geometry_signature == _neighbor_stencil_geometry_signature &&
      _neighbor_cell_num > 0 &&
      _d_neighbor_stencil_indices.size() ==
          static_cast<std::size_t>(_neighbor_cell_num) &&
      _linked_cell->_neighbor_cell.size() ==
          static_cast<std::size_t>(_linked_cell->_total_cells) *
              static_cast<std::size_t>(_neighbor_cell_num)) {
    return;
  }
  if (_linked_cell->_total_cells <= 0) {
    throw std::runtime_error("Invalid linked-cell grid for neighbor stencil");
  }

  const std::vector<rbmd::Id> stencil_indices =
      BuildNeighborStencilIndices(*_linked_cell);
  const rbmd::Id compact_neighbor_cell_num =
      static_cast<rbmd::Id>(stencil_indices.size());
  if (compact_neighbor_cell_num <= 0) {
    throw std::runtime_error("Empty linked-cell neighbor stencil");
  }
  const std::size_t required_size =
      static_cast<std::size_t>(_linked_cell->_total_cells) *
      static_cast<std::size_t>(compact_neighbor_cell_num);
  thrust::host_vector<rbmd::Id> host_stencil(stencil_indices.begin(),
                                              stencil_indices.end());
  _d_neighbor_stencil_indices = host_stencil;
  _neighbor_cell_num = compact_neighbor_cell_num;
  ComputeNeighborCells();
  if (_linked_cell->_neighbor_cell.size() != required_size) {
    throw std::runtime_error("Neighbor-cell stencil storage size mismatch");
  }
  Logger::Instance().info(
      "Neighbor stencil offsets: {} of {} (cell radius {})",
      compact_neighbor_cell_num, EffectiveNeighborCellNum(*_linked_cell),
      _linked_cell->_cell_count_within_cutoff);
  _neighbor_cell_grid_signature = signature;
  _neighbor_stencil_geometry_signature = geometry_signature;
  _neighbor_cells_valid = true;
}

void FullNeighborListBuilder::ComputeNeighborCells() {
  const std::size_t storage_size =
      static_cast<std::size_t>(_linked_cell->_total_cells) *
      static_cast<std::size_t>(this->_neighbor_cell_num);
  _linked_cell->_neighbor_cell.resize(storage_size);
  op::ComputeFullNeighborsOp<device::DEVICE_GPU> compute_full_neighbors_op;
  compute_full_neighbors_op(
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      thrust::raw_pointer_cast(_linked_cell->_cells.data()),
      this->_neighbor_cell_num, _linked_cell->_total_cells,
      _linked_cell->_cell_count_within_cutoff,
      thrust::raw_pointer_cast(_d_neighbor_stencil_indices.data()));
}
void FullNeighborListBuilder::ComputeNeighborCellsWithoutPBC() {
  const std::size_t storage_size =
      static_cast<std::size_t>(_linked_cell->_total_cells) *
      static_cast<std::size_t>(this->_neighbor_cell_num);
  _linked_cell->_neighbor_cell.resize(storage_size);
  op::ComputeFullNeighborsWithoutPBCOp<device::DEVICE_GPU>
      compute_full_neighbors_without_pbc_op;
  compute_full_neighbors_without_pbc_op(
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      this->_neighbor_cell_num, _linked_cell->_total_cells);
}

void FullNeighborListBuilder::EstimateNeighborsList() {
  Logger::Instance().debug("Reallocating neighbor list capacity");
  const rbmd::Id capacity =
      _required_neighbor_capacity > _neighbor_list->_max_neighbor_capacity
          ? GrownNeighborCapacity(_required_neighbor_capacity)
          : std::max(_neighbor_list->_max_neighbor_capacity,
                     InitialNeighborCapacity(
                         std::sqrt(_trunc_distance_power_2)));
  PrepareFixedWidthList(_neighbor_list, ActiveAtomCount(), capacity);
  _required_neighbor_capacity = 0;
  this->should_realloc = false;
}

rbmd::Id FullNeighborListBuilder::GenerateNeighborsList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  CHECK_RUNTIME(MEMCPY(_d_should_realloc, &(this->should_realloc),
                       sizeof(rbmd::Id), H2D));
  const rbmd::Id zero = 0;
  CHECK_RUNTIME(MEMCPY(
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()), &zero,
      sizeof(rbmd::Id), H2D));
  const rbmd::Id active_atoms = ActiveAtomCount();
  if (active_atoms == 0) {
    this->should_realloc = false;
    return RBMD_FALSE;
  }
  op::GenerateFullNeighborListOp<device::DEVICE_GPU>
      generate_full_neighbor_list_op;
  generate_full_neighbor_list_op(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _trunc_distance_power_2, active_atoms,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      _neighbor_cell_num,
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbor_num.data()),
      this->_neighbor_list->_max_neighbor_capacity,
      thrust::raw_pointer_cast(this->_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_end_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbors.data()),
      force_box, _d_should_realloc,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      ChooseThreadsPerParticle());
  CHECK_RUNTIME(MEMCPY(&(this->should_realloc), _d_should_realloc,
                       sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(MEMCPY(
      &_required_neighbor_capacity,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      sizeof(rbmd::Id), D2H));
  return this->should_realloc;
}
