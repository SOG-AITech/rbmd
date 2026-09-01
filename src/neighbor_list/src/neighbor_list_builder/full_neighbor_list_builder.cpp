
#include "neighbor_list_builder/full_neighbor_list_builder.h"

#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <output/include/Logger.hpp>
#include <sstream>
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

rbmd::Id RequestedNeighborCellNum(const LinkedCell& linked_cell) {
  return (2 * linked_cell._cell_count_within_cutoff + 1) *
         (2 * linked_cell._cell_count_within_cutoff + 1) *
         (2 * linked_cell._cell_count_within_cutoff + 1);
}

void EmitNeighborDebugCsv(const LinkedCell& linked_cell,
                          const NeighborList& neighbor_list,
                          const std::shared_ptr<DeviceData>& device_data) {
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
      stencil_cell_count = RequestedNeighborCellNum(linked_cell);
      if (stencil_cell_count > linked_cell._total_cells) {
        stencil_cell_count = linked_cell._total_cells;
      }
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
  this->_neighbor_cell_num = RequestedNeighborCellNum(*_linked_cell);
  this->_neighbor_list =
      std::make_shared<NeighborList>(_linked_cell->_native_atoms_num, false);
  this->_device_data = DataManager::getInstance().getDeviceData();
  _trunc_distance_power_2 =
      //_linked_cell->_cutoff * _linked_cell->_cutoff - EPSILON;
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff;
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
  this->_neighbor_cell_num = RequestedNeighborCellNum(*_linked_cell);
  _linked_cell->AssignAtomsToCell();
  _linked_cell->SortAtomsByCellKey();
  _linked_cell->ComputeCellRangesIndices();
  if (should_realloc) {
    // 好像就第一入口调用了  todo move to init?
    EstimateNeighborsList();
  }
  // 索引没有问题
  if (GenerateNeighborsList() == RBMD_TRUE) {
    EstimateNeighborsList();
    GenerateNeighborsList();
  }
  RecordNeighborBuildState(neighbor_cutoff);
#ifdef USE_MPI
  if (HasNeighborCache()) {
    if (!GET_RBMD_PARALLEL->PrepareForwardCoordinateExchange()) {
      InvalidateNeighborCache();
    }
  }
#endif
  EmitNeighborDebugCsv(*_linked_cell, *this->_neighbor_list, _device_data);
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
  this->_neighbor_cell_num = RequestedNeighborCellNum(*_linked_cell);
  _linked_cell->AssignAtomsToCell();
  _linked_cell->SortAtomsByCellKey();
  _linked_cell->ComputeCellRangesIndices();
  if (should_realloc) {
    // 好像就第一入口调用了  todo move to init?
    EstimateNeighborsList();
  }
  // 索引没有问题
  if (GenerateNeighborsList() == RBMD_TRUE) {
    EstimateNeighborsList();
    GenerateNeighborsList();
  }
  RecordNeighborBuildState(neighbor_cutoff);
#ifdef USE_MPI
  if (HasNeighborCache()) {
    if (!GET_RBMD_PARALLEL->PrepareForwardCoordinateExchange()) {
      InvalidateNeighborCache();
    }
  }
#endif
  EmitNeighborDebugCsv(*_linked_cell, *this->_neighbor_list, _device_data);
  return _neighbor_list;
}

void FullNeighborListBuilder::ComputeNeighborCells() {
  _linked_cell->_neighbor_cell.resize(
      (_linked_cell->_total_cells * this->_neighbor_cell_num));
  op::ComputeFullNeighborsOp<device::DEVICE_GPU> compute_full_neighbors_op;
  compute_full_neighbors_op(
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      thrust::raw_pointer_cast(_linked_cell->_cells.data()),
      this->_neighbor_cell_num, _linked_cell->_total_cells,
      _linked_cell->_cell_count_within_cutoff);
}
void FullNeighborListBuilder::ComputeNeighborCellsWithoutPBC() {
  _linked_cell->_neighbor_cell.resize(
      (_linked_cell->_total_cells * this->_neighbor_cell_num));
  op::ComputeFullNeighborsWithoutPBCOp<device::DEVICE_GPU>
      compute_full_neighbors_without_pbc_op;
  compute_full_neighbors_without_pbc_op(
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      this->_neighbor_cell_num, _linked_cell->_total_cells);
}

void FullNeighborListBuilder::EstimateNeighborsList() {
  Logger::Instance().debug("Reallocating neighbor list capacity");
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  rbmd::Id* d_total_max_neighbor_num;
  CHECK_RUNTIME(MALLOC(&d_total_max_neighbor_num, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MEMCPY(d_total_max_neighbor_num,
                       &(_neighbor_list->_h_total_max_neighbor_num),
                       sizeof(rbmd::Id), H2D));
  op::EstimateFullNeighborListOp<device::DEVICE_GPU>
      estimate_full_neighbor_list_op;
  estimate_full_neighbor_list_op(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _trunc_distance_power_2, active_atoms,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(
          this->_neighbor_list->_d_max_neighbor_num.data()),
      force_box, _linked_cell->GetDataPtr(), _neighbor_cell_num,
      _linked_cell->_cell_count_within_cutoff);
  ReductionSum(
      thrust::raw_pointer_cast(_neighbor_list->_d_max_neighbor_num.data()),
      d_total_max_neighbor_num, active_atoms);
  CHECK_RUNTIME(MEMCPY(&(_neighbor_list->_h_total_max_neighbor_num),
                       d_total_max_neighbor_num, sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(FREE(d_total_max_neighbor_num));
  _neighbor_list->_d_neighbors.resize(
      _neighbor_list->_h_total_max_neighbor_num);
  InitNeighborListIndices();
  this->should_realloc = false;
}

rbmd::Id FullNeighborListBuilder::GenerateNeighborsList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  CHECK_RUNTIME(MEMCPY(_d_should_realloc, &(this->should_realloc),
                       sizeof(rbmd::Id), H2D));
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
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
      thrust::raw_pointer_cast(
          this->_neighbor_list->_d_max_neighbor_num.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_end_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbors.data()),
      force_box, _d_should_realloc, _linked_cell->GetDataPtr(),
      _neighbor_cell_num, _linked_cell->_cell_count_within_cutoff);
  CHECK_RUNTIME(MEMCPY(&(this->should_realloc), _d_should_realloc,
                       sizeof(rbmd::Id), D2H));
  return this->should_realloc;
}
