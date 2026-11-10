#include "cvff.h"
#include "force_box_selector.h"

//#include <output/include/Logger.hpp>
#include <algorithm>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <limits>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "../../common/device_types.h"
#include "../../common/rbmd_define.h"
#include "../../common/types.h"
#include "../common/RBEPSample.h"
#include "../common/unit_factor.h"
#include "force_op/force_op.h"
#include "cvff_op/cvff_op.h"

#include "lj_op/lj_op.h"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"
#include "neighbor_list/include/neighbor_list_builder/full_neighbor_list_builder.h"
#include "neighbor_list/include/neighbor_list_builder/rbl_full_neighbor_list_builder.h"
#include "thrust/host_vector.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "thrust/transform_reduce.h"
#include "thrust/tuple.h"
// #include <hipcub/hipcub.hpp>
// #include <hipcub/backend/rocprim/block/block_reduce.hpp>
#include "common/math_utils.h"
#include "common/mpi_reduce_helper.hpp"
#include "common/mpi_root_guard.hpp"
#include "common/thermo_stats.hpp"
#include "rbsog_level_preset.h"
#include "common/timing_statistics.hpp"
extern rbmd::Id test_current_step;
extern std::map<std::string, UNIT> unit_factor_map;

namespace {
namespace fs = std::experimental::filesystem;

struct ForceSum3 {
  rbmd::Real x;
  rbmd::Real y;
  rbmd::Real z;
};

struct MakeForceSum3 {
  template <typename Tuple>
  __host__ __device__ ForceSum3 operator()(const Tuple& values) const {
    return {thrust::get<0>(values), thrust::get<1>(values),
            thrust::get<2>(values)};
  }
};

struct AddForceSum3 {
  __host__ __device__ ForceSum3 operator()(const ForceSum3& lhs,
                                           const ForceSum3& rhs) const {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
  }
};

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

bool IsMultiRankMpiRun() {
#ifdef USE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return false;
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized != 0) {
    return false;
  }
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size > 1;
#else
  return false;
#endif
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

struct TrackedSpecialDebugConfig {
  bool enabled{false};
  int every{1};
  std::vector<rbmd::Id> gids{};
};

TrackedSpecialDebugConfig ParseTrackedSpecialDebugConfig() {
  TrackedSpecialDebugConfig cfg;
  cfg.enabled = EnvFlagEnabled("RBMD_DEBUG_NEIGHBOR_STATS");
  cfg.every = EnvIntOrDefault("RBMD_DEBUG_NEIGHBOR_STATS_EVERY", 1);
  cfg.gids = ParseTrackedGidListFromEnv("RBMD_DEBUG_TRACK_GIDS");
  if (cfg.gids.empty()) {
    cfg.gids = ParseTrackedGidListFromEnv("RBMD_DEBUG_EXCHANGE_ROUTE_GIDS");
  }
  return cfg;
}

const TrackedSpecialDebugConfig& GetTrackedSpecialDebugConfig() {
  static const TrackedSpecialDebugConfig cfg = ParseTrackedSpecialDebugConfig();
  return cfg;
}

bool ShouldWriteTrackedSpecialDebug(rbmd::Id step) {
  const auto& cfg = GetTrackedSpecialDebugConfig();
  return cfg.enabled && cfg.every > 0 && (step % cfg.every) == 0 &&
         !cfg.gids.empty();
}

bool ShouldWriteForceStageDebug(rbmd::Id step) {
  const auto& cfg = GetTrackedSpecialDebugConfig();
  return cfg.enabled && cfg.every > 0 && (step % cfg.every) == 0;
}

bool ForceStageSyncEnabled() {
  static const bool enabled = EnvFlagEnabled("RBMD_DEBUG_FORCE_STAGE_SYNC");
  return enabled;
}

void DebugSyncForceStage(const char* stage) {
  if (!ForceStageSyncEnabled() || !stage || stage[0] == '\0') {
    return;
  }

  Logger::Instance().info("[force_stage_sync] step={} stage={} before_sync",
                          test_current_step, stage);
  CHECK_RUNTIME(DEVICESYNC());
  Logger::Instance().info("[force_stage_sync] step={} stage={} after_sync",
                          test_current_step, stage);
}

using DeviceTimingClock = std::chrono::high_resolution_clock;

DeviceTimingClock::time_point StartDeviceTiming() {
  CHECK_RUNTIME(DEVICESYNC());
  return DeviceTimingClock::now();
}

void RecordDeviceTiming(const char* category,
                        DeviceTimingClock::time_point start) {
  CHECK_RUNTIME(DEVICESYNC());
  const auto end = DeviceTimingClock::now();
  const std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record(category, duration.count());
}

bool DetailedPairTimingEnabled() {
  static const bool enabled = EnvFlagEnabled("RBMD_DETAILED_PAIR_TIMING");
  return enabled;
}

DeviceTimingClock::time_point StartDetailedDeviceTiming() {
  return DetailedPairTimingEnabled() ? StartDeviceTiming()
                                     : DeviceTimingClock::now();
}

void RecordDetailedDeviceTiming(const char* category,
                                DeviceTimingClock::time_point start) {
  if (DetailedPairTimingEnabled()) {
    RecordDeviceTiming(category, start);
  }
}

void RecordDetailedHostTiming(const char* category,
                              DeviceTimingClock::time_point start) {
  if (!DetailedPairTimingEnabled()) {
    return;
  }
  const std::chrono::duration<double> duration = DeviceTimingClock::now() - start;
  TimingStatistics::Instance().record(category, duration.count());
}

void DebugForceStageMarker(const char* stage) {
  if (!ForceStageSyncEnabled() || !stage || stage[0] == '\0') {
    return;
  }
  Logger::Instance().info("[force_stage_marker] step={} stage={}",
                          test_current_step, stage);
}

int CurrentProcessId() {
#if defined(_WIN32)
  return static_cast<int>(_getpid());
#else
  return static_cast<int>(::getpid());
#endif
}

int CurrentRank() {
#ifdef USE_MPI
  if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
    return GET_RBMD_PARALLEL->_domdec->_current_rank;
  }
#endif
  return 0;
}

void EnsureDebugLogDirectory() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  try {
    fs::create_directories("logs/debug");
  } catch (...) {
  }
  initialized = true;
}

std::string SafeCurrentPath() {
  try {
    return fs::current_path().string();
  } catch (const std::exception&) {
    return "<unknown>";
  }
}

bool OpenOutputFile(std::ofstream& file, const std::string& path,
                    std::ios::openmode mode) {
  if (!rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    return false;
  }
  try {
    const auto status = fs::symlink_status(path);
    if (fs::exists(status) && !fs::is_regular_file(status)) {
      Logger::Instance().error(
          "Refuse to open output file '{}' in cwd '{}': target is not a regular file.",
          path, SafeCurrentPath());
      return false;
    }
  } catch (const std::exception&) {
    Logger::Instance().error(
        "Failed to inspect output target '{}' in cwd '{}'.", path,
        SafeCurrentPath());
    return false;
  }

  file.open(path, mode);
  if (!file.is_open()) {
    Logger::Instance().error("Failed to open output file '{}' in cwd '{}'.", path,
                             SafeCurrentPath());
    return false;
  }
  return true;
}

struct SpecialBondWeights {
  rbmd::Real w12{1.0};
  rbmd::Real w13{1.0};
  rbmd::Real w14{1.0};
};

SpecialBondWeights read_special_bond_weights() {
  SpecialBondWeights weights;
  try {
    if (auto cfg = DataManager::getInstance().getConfigData()) {
      auto values =
          cfg->GetArray<rbmd::Real>("special_bonds", "hyper_parameters",
                                    "extend");
      if (!values.empty()) {
        weights.w12 = values[0];
        if (values.size() > 1) {
          weights.w13 = values[1];
        }
        if (values.size() > 2) {
          weights.w14 = values[2];
        }
      }
    }
  } catch (...) {
    // Keep the default LAMMPS-compatible weights when config is absent.
  }
  return weights;
}

void EmitTrackedSpecialDebugCsv(
    const std::shared_ptr<DeviceData>& device_data,
    const std::shared_ptr<NeighborList>& neighbor_list,
    const std::shared_ptr<LinkedCell>& linked_cell,
    rbmd::Id nlocal) {
  if (!device_data || !neighbor_list || !linked_cell ||
      !ShouldWriteTrackedSpecialDebug(test_current_step)) {
    return;
  }

  const auto& cfg = GetTrackedSpecialDebugConfig();
  const rbmd::Id safe_nlocal = std::max<rbmd::Id>(nlocal, 0);
  const rbmd::Id safe_ntotal = std::max<rbmd::Id>(linked_cell->_total_atoms_num, 0);
  const std::size_t native_size = static_cast<std::size_t>(safe_nlocal);
  const std::size_t total_size = static_cast<std::size_t>(safe_ntotal);

  const std::size_t ids_size =
      std::min(total_size, device_data->_d_atoms_id.size());
  const std::size_t start_size =
      std::min(native_size, neighbor_list->_start_idx.size());
  const std::size_t end_size =
      std::min(native_size, neighbor_list->_end_idx.size());
  const std::size_t nspecial_rows =
      std::min(native_size, device_data->d_nspecial.size() / 3);
  const std::size_t special_rows =
      (device_data->maxspecial > 0)
          ? std::min(native_size,
                     device_data->d_special.size() /
                         static_cast<std::size_t>(device_data->maxspecial))
          : 0;

  thrust::host_vector<rbmd::Id> h_ids(device_data->_d_atoms_id.begin(),
                                      device_data->_d_atoms_id.begin() + ids_size);
  thrust::host_vector<rbmd::Id> h_start(
      neighbor_list->_start_idx.begin(),
      neighbor_list->_start_idx.begin() + start_size);
  thrust::host_vector<rbmd::Id> h_end(
      neighbor_list->_end_idx.begin(),
      neighbor_list->_end_idx.begin() + end_size);
  thrust::host_vector<rbmd::Id> h_neighbors = neighbor_list->_d_neighbors;
  thrust::host_vector<int> h_nspecial;
  thrust::host_vector<rbmd::Id> h_special;
  if (nspecial_rows > 0) {
    h_nspecial = thrust::host_vector<int>(
        device_data->d_nspecial.begin(),
        device_data->d_nspecial.begin() + nspecial_rows * 3);
  }
  if (special_rows > 0 && device_data->maxspecial > 0) {
    h_special = thrust::host_vector<rbmd::Id>(
        device_data->d_special.begin(),
        device_data->d_special.begin() +
            special_rows * static_cast<std::size_t>(device_data->maxspecial));
  }

  std::unordered_map<rbmd::Id, std::vector<rbmd::Id>> gid_to_indices;
  gid_to_indices.reserve(cfg.gids.size());
  for (std::size_t idx = 0; idx < native_size && idx < h_ids.size(); ++idx) {
    gid_to_indices[h_ids[idx]].push_back(static_cast<rbmd::Id>(idx));
  }

  EnsureDebugLogDirectory();
  const int rank = CurrentRank();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/special_gid_stats_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,nlocal,ntotal,gid,local_count,local_idx,neighbor_count,"
           "special_hit_count,special_hit_12,special_hit_13,special_hit_14,"
           "n12,n13,n14\n";
  }

  auto clamp_offset = [&](rbmd::Id raw) -> rbmd::Id {
    if (raw < 0) {
      return 0;
    }
    const auto capped = static_cast<std::size_t>(raw);
    return static_cast<rbmd::Id>(
        capped > h_neighbors.size() ? h_neighbors.size() : capped);
  };

  for (const rbmd::Id gid : cfg.gids) {
    const auto it = gid_to_indices.find(gid);
    const rbmd::Id local_count =
        (it == gid_to_indices.end())
            ? rbmd::Id(0)
            : static_cast<rbmd::Id>(it->second.size());
    const rbmd::Id local_idx =
        (it == gid_to_indices.end() || it->second.empty()) ? rbmd::Id(-1)
                                                           : it->second.front();

    rbmd::Id neighbor_count = -1;
    int n12 = -1;
    int n13 = -1;
    int n14 = -1;
    rbmd::Id special_hit_count = -1;
    rbmd::Id special_hit_12 = -1;
    rbmd::Id special_hit_13 = -1;
    rbmd::Id special_hit_14 = -1;

    if (local_idx >= 0 && static_cast<std::size_t>(local_idx) < start_size &&
        static_cast<std::size_t>(local_idx) < end_size) {
      const rbmd::Id safe_start =
          clamp_offset(h_start[static_cast<std::size_t>(local_idx)]);
      const rbmd::Id safe_end =
          clamp_offset(h_end[static_cast<std::size_t>(local_idx)]);
      const rbmd::Id begin = std::min(safe_start, safe_end);
      const rbmd::Id end = std::max(safe_start, safe_end);
      neighbor_count = end - begin;
      special_hit_count = 0;
      special_hit_12 = 0;
      special_hit_13 = 0;
      special_hit_14 = 0;

      if (static_cast<std::size_t>(local_idx) < nspecial_rows) {
        const std::size_t base = static_cast<std::size_t>(local_idx) * 3;
        n12 = h_nspecial[base + 0];
        n13 = h_nspecial[base + 1];
        n14 = h_nspecial[base + 2];
      } else {
        n12 = 0;
        n13 = 0;
        n14 = 0;
      }

      const int limit = std::max(0, std::min(n14, device_data->maxspecial));
      std::unordered_map<rbmd::Id, int> special_band;
      if (device_data->maxspecial > 0 &&
          static_cast<std::size_t>(local_idx) < special_rows) {
        const std::size_t row_base =
            static_cast<std::size_t>(local_idx) *
            static_cast<std::size_t>(device_data->maxspecial);
        for (int slot = 0; slot < limit; ++slot) {
          const rbmd::Id partner = h_special[row_base + static_cast<std::size_t>(slot)];
          if (partner <= 0) {
            continue;
          }
          int band = 14;
          if (slot < n12) {
            band = 12;
          } else if (slot < n13) {
            band = 13;
          }
          special_band.emplace(partner, band);
        }
      }

      for (rbmd::Id slot = begin; slot < end; ++slot) {
        const std::size_t offset = static_cast<std::size_t>(slot);
        if (offset >= h_neighbors.size()) {
          continue;
        }
        const rbmd::Id neighbor_idx = h_neighbors[offset];
        if (neighbor_idx < 0 ||
            static_cast<std::size_t>(neighbor_idx) >= h_ids.size()) {
          continue;
        }
        const rbmd::Id neighbor_gid = h_ids[static_cast<std::size_t>(neighbor_idx)];
        const auto band_it = special_band.find(neighbor_gid);
        if (band_it == special_band.end()) {
          continue;
        }
        ++special_hit_count;
        if (band_it->second == 12) {
          ++special_hit_12;
        } else if (band_it->second == 13) {
          ++special_hit_13;
        } else {
          ++special_hit_14;
        }
      }
    }

    csv << test_current_step << "," << rank << "," << safe_nlocal << ","
        << safe_ntotal << "," << gid << "," << local_count << "," << local_idx
        << "," << neighbor_count << "," << special_hit_count << ","
        << special_hit_12 << "," << special_hit_13 << "," << special_hit_14
        << "," << n12 << "," << n13 << "," << n14 << "\n";
  }
}

void EmitBadForceNeighborCsv(
    const std::shared_ptr<DeviceData>& device_data,
    const std::shared_ptr<NeighborList>& neighbor_list,
    const std::shared_ptr<LinkedCell>& linked_cell, const Box& force_box,
    const char* stage, rbmd::Id nlocal, rbmd::Id atom_idx,
    rbmd::Id atom_gid) {
  if (!device_data || !neighbor_list || !linked_cell || !stage ||
      stage[0] == '\0' || atom_idx < 0) {
    return;
  }

  const rbmd::Id safe_nlocal = std::max<rbmd::Id>(nlocal, 0);
  const rbmd::Id safe_ntotal = std::max<rbmd::Id>(linked_cell->_total_atoms_num, 0);
  const std::size_t native_size = std::min<std::size_t>(
      static_cast<std::size_t>(safe_nlocal),
      std::min(neighbor_list->_start_idx.size(), neighbor_list->_end_idx.size()));
  const std::size_t total_size = std::min<std::size_t>(
      static_cast<std::size_t>(safe_ntotal),
      std::min(
          device_data->_d_atoms_id.size(),
          std::min(
              device_data->_d_atoms_type.size(),
              std::min(
                  device_data->_d_charge.size(),
                  std::min(
                      device_data->_d_px.size(),
                      std::min(
                          device_data->_d_py.size(),
                          std::min(device_data->_d_pz.size(),
                                   linked_cell->_per_atom_cell_id.size())))))));

  if (static_cast<std::size_t>(atom_idx) >= native_size ||
      static_cast<std::size_t>(atom_idx) >= total_size) {
    return;
  }

  thrust::host_vector<rbmd::Id> h_ids(device_data->_d_atoms_id.begin(),
                                      device_data->_d_atoms_id.begin() + total_size);
  thrust::host_vector<rbmd::Id> h_types(
      device_data->_d_atoms_type.begin(),
      device_data->_d_atoms_type.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_charge(
      device_data->_d_charge.begin(), device_data->_d_charge.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_px(device_data->_d_px.begin(),
                                       device_data->_d_px.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_py(device_data->_d_py.begin(),
                                       device_data->_d_py.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_pz(device_data->_d_pz.begin(),
                                       device_data->_d_pz.begin() + total_size);
  thrust::host_vector<rbmd::Id> h_cell_id(
      linked_cell->_per_atom_cell_id.begin(),
      linked_cell->_per_atom_cell_id.begin() + total_size);
  thrust::host_vector<rbmd::Id> h_start(
      neighbor_list->_start_idx.begin(),
      neighbor_list->_start_idx.begin() + native_size);
  thrust::host_vector<rbmd::Id> h_end(neighbor_list->_end_idx.begin(),
                                      neighbor_list->_end_idx.begin() + native_size);
  thrust::host_vector<rbmd::Id> h_neighbors = neighbor_list->_d_neighbors;
  thrust::host_vector<Cell> h_cells = linked_cell->_cells;

  const auto clamp_offset = [&](rbmd::Id raw) -> rbmd::Id {
    if (raw < 0) {
      return 0;
    }
    const auto capped = static_cast<std::size_t>(raw);
    return static_cast<rbmd::Id>(
        capped > h_neighbors.size() ? h_neighbors.size() : capped);
  };

  const rbmd::Id self_cell_idx = h_cell_id[static_cast<std::size_t>(atom_idx)];
  int self_cell_is_halo = -1;
  if (self_cell_idx >= 0 &&
      static_cast<std::size_t>(self_cell_idx) < h_cells.size()) {
    self_cell_is_halo =
        h_cells[static_cast<std::size_t>(self_cell_idx)]._is_halo ? 1 : 0;
  }
  const rbmd::Real self_x = h_px[static_cast<std::size_t>(atom_idx)];
  const rbmd::Real self_y = h_py[static_cast<std::size_t>(atom_idx)];
  const rbmd::Real self_z = h_pz[static_cast<std::size_t>(atom_idx)];
  const rbmd::Real self_charge = h_charge[static_cast<std::size_t>(atom_idx)];
  const rbmd::Id self_type = h_types[static_cast<std::size_t>(atom_idx)];

  const rbmd::Id begin =
      clamp_offset(h_start[static_cast<std::size_t>(atom_idx)]);
  const rbmd::Id end = clamp_offset(h_end[static_cast<std::size_t>(atom_idx)]);

  EnsureDebugLogDirectory();
  const int rank = CurrentRank();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/bad_force_neighbor_stats_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::out | std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,stage,nlocal,atom_idx,atom_gid,atom_cell_idx,"
           "atom_cell_is_halo,slot,neighbor_idx,neighbor_gid,neighbor_region,"
           "neighbor_cell_idx,neighbor_cell_is_halo,same_gid,distance_sq,"
           "dx,dy,dz,self_charge,neighbor_charge,self_type,neighbor_type\n";
  }

  for (rbmd::Id slot = begin; slot < end; ++slot) {
    const std::size_t offset = static_cast<std::size_t>(slot);
    if (offset >= h_neighbors.size()) {
      continue;
    }

    const rbmd::Id neighbor_idx = h_neighbors[offset];
    if (neighbor_idx < 0 ||
        static_cast<std::size_t>(neighbor_idx) >= total_size) {
      continue;
    }

    const rbmd::Id neighbor_gid = h_ids[static_cast<std::size_t>(neighbor_idx)];
    const char* neighbor_region =
        (neighbor_idx < safe_nlocal) ? "native" : "ghost";
    const rbmd::Id neighbor_cell_idx =
        h_cell_id[static_cast<std::size_t>(neighbor_idx)];
    int neighbor_cell_is_halo = -1;
    if (neighbor_cell_idx >= 0 &&
        static_cast<std::size_t>(neighbor_cell_idx) < h_cells.size()) {
      neighbor_cell_is_halo =
          h_cells[static_cast<std::size_t>(neighbor_cell_idx)]._is_halo ? 1 : 0;
    }

    rbmd::Real dx =
        h_px[static_cast<std::size_t>(neighbor_idx)] - self_x;
    rbmd::Real dy =
        h_py[static_cast<std::size_t>(neighbor_idx)] - self_y;
    rbmd::Real dz =
        h_pz[static_cast<std::size_t>(neighbor_idx)] - self_z;
    MinImageDistance_while(force_box, dx, dy, dz);
    const rbmd::Real distance_sq = dx * dx + dy * dy + dz * dz;

    csv << test_current_step << "," << rank << "," << stage << ","
        << safe_nlocal << "," << atom_idx << "," << atom_gid << ","
        << self_cell_idx << "," << self_cell_is_halo << "," << slot << ","
        << neighbor_idx << "," << neighbor_gid << "," << neighbor_region
        << "," << neighbor_cell_idx << "," << neighbor_cell_is_halo << ","
        << (neighbor_gid == atom_gid ? 1 : 0) << "," << distance_sq << ","
        << dx << "," << dy << "," << dz << "," << self_charge << ","
        << h_charge[static_cast<std::size_t>(neighbor_idx)] << ","
        << self_type << ","
        << h_types[static_cast<std::size_t>(neighbor_idx)] << "\n";
  }
}

void EmitForceStageDebugCsv(const std::shared_ptr<DeviceData>& device_data,
                            const char* stage,
                            const thrust::device_vector<rbmd::Real>& fx,
                            const thrust::device_vector<rbmd::Real>& fy,
                            const thrust::device_vector<rbmd::Real>& fz,
                            rbmd::Id nlocal,
                            const std::shared_ptr<NeighborList>& neighbor_list = nullptr,
                            const std::shared_ptr<LinkedCell>& linked_cell = nullptr,
                            Box force_box = Box()) {
  if (!device_data || !stage || stage[0] == '\0' ||
      !ShouldWriteForceStageDebug(test_current_step)) {
    return;
  }

  const rbmd::Id safe_nlocal = std::max<rbmd::Id>(nlocal, 0);
  const std::size_t sample_size = std::min<std::size_t>(
      static_cast<std::size_t>(safe_nlocal),
      std::min(fx.size(), std::min(fy.size(), fz.size())));
  const std::size_t ids_size =
      std::min<std::size_t>(sample_size, device_data->_d_atoms_id.size());

  thrust::host_vector<rbmd::Real> h_fx(fx.begin(), fx.begin() + sample_size);
  thrust::host_vector<rbmd::Real> h_fy(fy.begin(), fy.begin() + sample_size);
  thrust::host_vector<rbmd::Real> h_fz(fz.begin(), fz.begin() + sample_size);
  thrust::host_vector<rbmd::Id> h_ids(
      device_data->_d_atoms_id.begin(), device_data->_d_atoms_id.begin() + ids_size);

  rbmd::Id nan_x = 0;
  rbmd::Id nan_y = 0;
  rbmd::Id nan_z = 0;
  rbmd::Id inf_x = 0;
  rbmd::Id inf_y = 0;
  rbmd::Id inf_z = 0;
  rbmd::Id first_bad_idx = -1;
  rbmd::Id first_bad_gid = -1;
  rbmd::Id first_bad_gid_occurrences = 0;
  rbmd::Id first_bad_gid_other_idx = -1;
  rbmd::Real max_abs_component = rbmd::Real(0);
  rbmd::Id max_abs_idx = -1;
  rbmd::Id max_abs_gid = -1;

  auto note_bad = [&](std::size_t idx) {
    if (first_bad_idx >= 0) {
      return;
    }
    first_bad_idx = static_cast<rbmd::Id>(idx);
    if (idx < h_ids.size()) {
      first_bad_gid = h_ids[idx];
    }
  };

  for (std::size_t idx = 0; idx < sample_size; ++idx) {
    const rbmd::Real values[3] = {h_fx[idx], h_fy[idx], h_fz[idx]};
    rbmd::Id* nan_counts[3] = {&nan_x, &nan_y, &nan_z};
    rbmd::Id* inf_counts[3] = {&inf_x, &inf_y, &inf_z};

    for (int axis = 0; axis < 3; ++axis) {
      const double v = static_cast<double>(values[axis]);
      if (std::isnan(v)) {
        ++(*nan_counts[axis]);
        note_bad(idx);
        continue;
      }
      if (!std::isfinite(v)) {
        ++(*inf_counts[axis]);
        note_bad(idx);
        continue;
      }

      const rbmd::Real abs_v = static_cast<rbmd::Real>(std::fabs(v));
      if (abs_v > max_abs_component) {
        max_abs_component = abs_v;
        max_abs_idx = static_cast<rbmd::Id>(idx);
        max_abs_gid = (idx < h_ids.size()) ? h_ids[idx] : rbmd::Id(-1);
      }
    }
  }

  if (first_bad_gid >= 0) {
    thrust::host_vector<rbmd::Id> h_all_ids = device_data->_d_atoms_id;
    for (std::size_t idx = 0; idx < h_all_ids.size(); ++idx) {
      if (h_all_ids[idx] != first_bad_gid) {
        continue;
      }
      ++first_bad_gid_occurrences;
      if (static_cast<rbmd::Id>(idx) != first_bad_idx &&
          first_bad_gid_other_idx < 0) {
        first_bad_gid_other_idx = static_cast<rbmd::Id>(idx);
      }
    }
  }

  EnsureDebugLogDirectory();
  const int rank = CurrentRank();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/force_stage_stats_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,stage,nlocal,nan_x,nan_y,nan_z,inf_x,inf_y,inf_z,"
           "first_bad_idx,first_bad_gid,first_bad_gid_occurrences,"
           "first_bad_gid_other_idx,max_abs_component,max_abs_idx,max_abs_gid\n";
  }
  csv << test_current_step << "," << rank << "," << stage << "," << safe_nlocal
      << "," << nan_x << "," << nan_y << "," << nan_z << "," << inf_x << ","
      << inf_y << "," << inf_z << "," << first_bad_idx << ","
      << first_bad_gid << "," << first_bad_gid_occurrences << ","
      << first_bad_gid_other_idx << "," << max_abs_component << ","
      << max_abs_idx << "," << max_abs_gid << "\n";

  if (first_bad_idx >= 0 && neighbor_list && linked_cell &&
      std::string(stage) == "after_short_range") {
    EmitBadForceNeighborCsv(device_data, neighbor_list, linked_cell, force_box,
                            stage, safe_nlocal, first_bad_idx, first_bad_gid);
  }
}
}  // namespace

CVFF::CVFF()
{
  _rbl_neighbor_list_builder = std::make_shared<RblFullNeighborListBuilder>();
  _neighbor_list_builder = std::make_shared<FullNeighborListBuilder>();

  _kspace_calculator = std::make_unique<KSpaceCalculator>();

  auto unit = DataManager::getInstance().getConfigData()->Get
<std::string>("unit", "init_configuration", "read_data");
  UNIT unit_factor = ParseUnit(unit);
  switch (unit_factor) {
    case UNIT::LJ:
      _qqr2e = UnitFactor<UNIT::LJ>::_qqr2e;
      break;
    case UNIT::METAL:
      _qqr2e = UnitFactor<UNIT::METAL>::_qqr2e;
      break;
    case UNIT::REAL:
      _qqr2e = UnitFactor<UNIT::REAL>::_qqr2e;
      break;

    default:
      break;
  }

  if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    std::remove("thermo.txt");
  }
}

CVFF::~CVFF(){}

void CVFF::Init()
{
  const auto& config = DataManager::getInstance().getConfigData();

  //neighbor
  _cut_off = config->Get<rbmd::Real>("cut_off", "hyper_parameters", "neighbor");
  _neighbor_type = config->Get<std::string>("type", "hyper_parameters", "neighbor");

  if("RBL" == _neighbor_type) {
    bool energy_rbl_flag = config->PathExists({"hyper_parameters", "neighbor" ,"energy_rbl_flag"});
    if (energy_rbl_flag) {
      _energy_rbl_flag = config->Get<std::string>("energy_rbl_flag", "hyper_parameters", "neighbor");
    }
    else {
      Logger::Instance().error( "\033[31m When using RBL for the neighbor type, "
                   "the key 'energy_rbl_flag' must be defined.\033[0m");
      exit(EXIT_FAILURE); //
    }
  }
  else if("VERLET-SOG" == _neighbor_type) {
    LJCoulSOGInit();
  }
  //
  _kspace_calculator->Init();
  _alpha = _kspace_calculator->GetAlpha();

  _lj_num_types = *(_structure_info_data->_num_atoms_type);
  if (_lj_num_types <= 0) {
    throw std::runtime_error("CVFF requires at least one LJ atom type");
  }
  thrust::host_vector<rbmd::Real> h_eps = _device_data->_d_eps;
  thrust::host_vector<rbmd::Real> h_sigma = _device_data->_d_sigma;
  const std::size_t pair_table_size =
      static_cast<std::size_t>(_lj_num_types) *
      static_cast<std::size_t>(_lj_num_types);
  thrust::host_vector<rbmd::Real> h_lj_force12(pair_table_size);
  thrust::host_vector<rbmd::Real> h_lj_force6(pair_table_size);
  for (rbmd::Id type_i = 0; type_i < _lj_num_types; ++type_i) {
    for (rbmd::Id type_j = 0; type_j < _lj_num_types; ++type_j) {
      const rbmd::Real epsilon =
          SQRT(h_eps[type_i] * h_eps[type_j]);
      const rbmd::Real sigma =
          (h_sigma[type_i] + h_sigma[type_j]) * rbmd::Real(0.5);
      const rbmd::Real sigma2 = sigma * sigma;
      const rbmd::Real sigma6 = sigma2 * sigma2 * sigma2;
      const std::size_t index =
          static_cast<std::size_t>(type_i) * _lj_num_types + type_j;
      h_lj_force12[index] = rbmd::Real(48) * epsilon * sigma6 * sigma6;
      h_lj_force6[index] = rbmd::Real(24) * epsilon * sigma6;
    }
  }
  _d_lj_force12 = h_lj_force12;
  _d_lj_force6 = h_lj_force6;
  _global_num_atoms = GetGlobalIdSum(*(_structure_info_data->_num_atoms));
  if (_global_num_atoms <= 0) {
    throw std::runtime_error("CVFF global atom count must be positive");
  }

  if ("VERLET-SOG" == _neighbor_type) {
    Logger::Instance().info(
        "{} initialization ...\n"
        "         cut_off : {}\n",
        _neighbor_type, _cut_off);
  } else {
    Logger::Instance().info(
        "{} initialization ...\n"
        "         cut_off : {}\n"
        "         alpha : {}\n",
        _neighbor_type, _cut_off, _alpha);
  }
}

bool CVFF::UsePerAtomBondPath() const {
#ifdef USE_MPI
  if (!IsMultiRankMpiRun() || !_device_data) {
    return false;
  }

  return _device_data->bond_per_atom > 0 && !_device_data->d_num_bond.empty() &&
         !_device_data->d_bond_type.empty() &&
         !_device_data->d_bond_atom.empty();
#else
  return false;
#endif
}

bool CVFF::UsePerAtomAnglePath() const {
#ifdef USE_MPI
  if (!IsMultiRankMpiRun() || !_device_data) {
    return false;
  }

  return _device_data->angle_per_atom > 0 && !_device_data->d_num_angle.empty() &&
         !_device_data->d_angle_type.empty() &&
         !_device_data->d_angle_atom1.empty() &&
         !_device_data->d_angle_atom2.empty() &&
         !_device_data->d_angle_atom3.empty();
#else
  return false;
#endif
}

bool CVFF::UsePerAtomDihedralPath() const {
#ifdef USE_MPI
  if (!IsMultiRankMpiRun() || !_device_data) {
    return false;
  }

  return _device_data->dihedral_per_atom > 0 &&
         !_device_data->d_num_dihedral.empty() &&
         !_device_data->d_dihedral_type.empty() &&
         !_device_data->d_dihedral_atom1.empty() &&
         !_device_data->d_dihedral_atom2.empty() &&
         !_device_data->d_dihedral_atom3.empty() &&
         !_device_data->d_dihedral_atom4.empty();
#else
  return false;
#endif
}

bool CVFF::UsePerAtomImproperPath() const {
#ifdef USE_MPI
  if (!IsMultiRankMpiRun() || !_device_data) {
    return false;
  }

  const bool has_positive_limit = _device_data->improper_per_atom > 0;
  const bool has_required_arrays =
      !_device_data->d_num_improper.empty() &&
      !_device_data->d_improper_type.empty() &&
      !_device_data->d_improper_atom1.empty() &&
      !_device_data->d_improper_atom2.empty() &&
      !_device_data->d_improper_atom3.empty() &&
      !_device_data->d_improper_atom4.empty();
  return has_positive_limit && has_required_arrays;
#else
  return false;
#endif
}

void CVFF::LJCoulSOGInit() {

   const auto& config = DataManager::getInstance().getConfigData();
   const auto raw_level =
       config->Get<rbmd::Id>("rbsog_level", "hyper_parameters", "coulomb");
   const auto level = static_cast<long long>(raw_level);
   const auto& presets = GetRBSOGLevelPresets();
   if (level < 0 || level >= static_cast<long long>(presets.size())) {
     Logger::Instance().error(
         "\033[31m RBSOG rbsog_level must be in [0, 4]. Current value: {}.\033[0m",
         raw_level);
     exit(EXIT_FAILURE);
   }

   const auto& preset = presets[static_cast<std::size_t>(level)];
   const rbmd::Real sigma_at_rc10 = preset.sigma_at_rc10;
   _rbsog_b = preset.b;
   _rbsog_sigma = sigma_at_rc10 * _cut_off / 10.0;
   _rbsog_Mmax = preset.mmax;


  rbmd::Real r0 = _cut_off / _rbsog_sigma;
  _w0 = Compute_W01(r0, _rbsog_b);

  std::vector<rbmd::Real> bl;
  bl.resize(_rbsog_Mmax);
  std::vector<rbmd::Real> bl3_inv;
  bl3_inv.resize(_rbsog_Mmax);
  std::vector<rbmd::Real> BL2SIGMA2INV;
  BL2SIGMA2INV.resize(_rbsog_Mmax);

  for (int i = 0; i < _rbsog_Mmax; i++)
  {
    bl[i] = POW(_rbsog_b, i);
    rbmd::Real bl_inv = 1.0 / bl[i];
    bl3_inv[i] = bl_inv * bl_inv * bl_inv;
    BL2SIGMA2INV[i] = 1.0 / (2.0 * _rbsog_sigma * _rbsog_sigma * bl[i] * bl[i]);
  }
  bl3_inv[0] = _w0;

  rbmd::Real coef = LOG(_rbsog_b) / (_rbsog_sigma * _rbsog_sigma *
                  SQRT(2 * M_PI * _rbsog_sigma * _rbsog_sigma));

  std::vector<rbmd::Real>TaylorCoeff;
  TaylorCoeff.resize(6);

  for (int i = 0; i < TaylorCoeff.size(); i++)
  {
    double sumsum = 0.00;
    for (int j = 0; j < _rbsog_Mmax; j++)
    {
      sumsum = sumsum + bl3_inv[j] * (1.0 / MathLib::factorial(i+0.00)) * POW(BL2SIGMA2INV[j], i+0.00);
    }
    TaylorCoeff[i] = POW(-1.0,i+1.0) * 2.0 * coef * sumsum;
  }
  for (rbmd::Id i = 0; i < 6; ++i)
  {
    std::cout << "TaylorCoeff on rbmd-sog: " << TaylorCoeff[i] << std::endl;
  }
  _d_taylor_coeff = TaylorCoeff;
}

void CVFF::Execute() {
  const auto& config = DataManager::getInstance().getConfigData();
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);

  ComputeLJCutCoulForce();
  DebugSyncForceStage("after_short_range_kernel");
  EmitForceStageDebugCsv(_device_data, "after_short_range",
                         _device_data->_d_force_ljcoul_x,
                         _device_data->_d_force_ljcoul_y,
                         _device_data->_d_force_ljcoul_z,
                         *(_structure_info_data->_num_atoms), _list,
                         linked_cell, force_box);
  if(config->PathExists({"hyper_parameters", "coulomb"})) {
    _kspace_calculator->Execute();
    DebugSyncForceStage("after_kspace_kernel");
    EmitForceStageDebugCsv(_device_data, "after_kspace",
                           _device_data->_d_force_kspace_x,
                           _device_data->_d_force_kspace_y,
                           _device_data->_d_force_kspace_z,
                           *(_structure_info_data->_num_atoms));
  }
  ComputeBondForce();
  DebugSyncForceStage("after_bond_kernel");
  ComputeAngleForce();
  DebugSyncForceStage("after_angle_kernel");
  if(*(_structure_info_data->_num_dihedrals)) {
    ComputeDihedralForce();
    DebugSyncForceStage("after_dihedral_kernel");
  }
  if(*(_structure_info_data->_num_impropers)) {
    ComputeImproperForce();
    DebugSyncForceStage("after_improper_kernel");
  }

  SumForces();
  DebugSyncForceStage("after_sum_forces_kernel");
  DebugForceStageMarker("before_emit_after_sum_forces_csv");
  EmitForceStageDebugCsv(_device_data, "after_sum_forces",
                         _device_data->_d_fx,
                         _device_data->_d_fy,
                         _device_data->_d_fz,
                         *(_structure_info_data->_num_atoms));
  DebugForceStageMarker("after_emit_after_sum_forces_csv");

  DebugForceStageMarker("before_evaluate_potential_energy");
  EvaluatePotentialEnergy();
  DebugForceStageMarker("after_evaluate_potential_energy");
}

void CVFF::ComputeLJCutCoulForce()
{
  if ("RBL" ==_neighbor_type)
  {
    ComputeLJRBL();
  }
  else if ("VERLET-SOG" ==_neighbor_type) {
    ComputeLJSOG();
  }
  else
  {
    ComputeLJVerlet();
  }

  //add thermo
  ThermoStats::Instance().AddThermoData("vdwl",_e_vdwl);
  ThermoStats::Instance().AddThermoData("coul",_e_coul);
}

void CVFF::ComputeLJRBL()
{
    // rbl_neighbor_list_build
    const auto start = StartDeviceTiming();
    _rbl_list = _rbl_neighbor_list_builder->Build();
    RecordDeviceTiming("Neighbor-List", start);

    // compute force
    const auto start_rbl_force = StartDeviceTiming();

    const auto r_core =
      DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
          "r_core", "hyper_parameters", "neighbor");

     const auto neighbor_sample_num =
     DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
         "neighbor_sample_num", "hyper_parameters", "neighbor");

    auto num_atoms = *(_structure_info_data->_num_atoms);
    const auto sbw = read_special_bond_weights();
    const int maxspecial = (_device_data && !_device_data->d_nspecial.empty() &&
                            !_device_data->d_special.empty())
                               ? _device_data->maxspecial
                               : 0;
    const int* d_nspecial =
        (!_device_data->d_nspecial.empty())
            ? thrust::raw_pointer_cast(_device_data->d_nspecial.data())
            : nullptr;
    const rbmd::Id* d_special =
        (!_device_data->d_special.empty())
            ? thrust::raw_pointer_cast(_device_data->d_special.data())
            : nullptr;
    const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
    const auto start_pair_kernel = StartDetailedDeviceTiming();
    op::SpecialLJCutCoulRBLForceOp<device::DEVICE_GPU>()(
        force_box,r_core, _cut_off, num_atoms,neighbor_sample_num,
      _rbl_list->_selection_frequency,_alpha,_qqr2e,
        thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
        thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
        _lj_num_types,
        thrust::raw_pointer_cast(_d_lj_force12.data()),
        thrust::raw_pointer_cast(_d_lj_force6.data()),
        thrust::raw_pointer_cast(_rbl_list->_start_idx.data()),
        thrust::raw_pointer_cast(_rbl_list->_end_idx.data()),
        thrust::raw_pointer_cast(_rbl_list->_d_neighbors.data()),
        thrust::raw_pointer_cast(_rbl_list->_d_random_neighbor.data()),
        thrust::raw_pointer_cast(_rbl_list->_d_random_neighbor_num.data()),
        d_nspecial,
        d_special,
        maxspecial,
        sbw.w12,
        sbw.w13,
        sbw.w14,
        thrust::raw_pointer_cast(_device_data->_d_charge.data()),
        thrust::raw_pointer_cast(_device_data->_d_px.data()),
        thrust::raw_pointer_cast(_device_data->_d_py.data()),
        thrust::raw_pointer_cast(_device_data->_d_pz.data()),
        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()));
    RecordDetailedDeviceTiming("RBL-PairKernel", start_pair_kernel);

    const auto start_local_reduction = StartDetailedDeviceTiming();
    size_t reduce_span = static_cast<size_t>(num_atoms > 0 ? num_atoms : 0);
    if (reduce_span > _device_data->_d_force_ljcoul_x.size()) {
      reduce_span = _device_data->_d_force_ljcoul_x.size();
    }
    const auto force_begin = thrust::make_zip_iterator(thrust::make_tuple(
        _device_data->_d_force_ljcoul_x.begin(),
        _device_data->_d_force_ljcoul_y.begin(),
        _device_data->_d_force_ljcoul_z.begin()));
    const ForceSum3 local_sum = thrust::transform_reduce(
        force_begin, force_begin + reduce_span, MakeForceSum3{},
        ForceSum3{rbmd::Real(0), rbmd::Real(0), rbmd::Real(0)},
        AddForceSum3{});
    RecordDetailedDeviceTiming("RBL-LocalReduction", start_local_reduction);

    const auto start_mpi_reduction = DeviceTimingClock::now();
    rbmd::Real global_sum[3] = {local_sum.x, local_sum.y, local_sum.z};
    AllReduceRealBufferInPlace(global_sum, 3);
    _corr_value_x = global_sum[0] / _global_num_atoms;
    _corr_value_y = global_sum[1] / _global_num_atoms;
    _corr_value_z = global_sum[2] / _global_num_atoms;
    RecordDetailedHostTiming("RBL-MPIAllreduce", start_mpi_reduction);

    // fix RBL:   rbl_force = force - corr_value
    const auto start_correction = StartDetailedDeviceTiming();
    op::FixRBLForceOp<device::DEVICE_GPU>()(
                        num_atoms, _corr_value_x, _corr_value_y, _corr_value_z,
                        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
                        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
                        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()));
    RecordDetailedDeviceTiming("RBL-CorrectionApply", start_correction);

  RecordDeviceTiming("Short-Range", start_rbl_force);

    //energy
    if ("yes" == _energy_rbl_flag ) {
      ComputeLJCoulEnergy();
    }
}

void CVFF::ComputeLJVerlet()
{
  //neighbor_list_build
  const auto start = StartDeviceTiming();
  _list = _neighbor_list_builder->Build();
  RecordDeviceTiming("Neighbor-List", start);

  //
  const auto start_verlet_force = StartDeviceTiming();
  //
  thrust::device_vector<rbmd::Real> _d_total_evdwl(1, 0.0);
  thrust::device_vector<rbmd::Real> _d_total_ecoul(1, 0.0);
  //
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  EmitTrackedSpecialDebugCsv(_device_data, _list, linked_cell, num_atoms);
  const auto sbw = read_special_bond_weights();
  const int maxspecial = (_device_data && !_device_data->d_nspecial.empty() &&
                          !_device_data->d_special.empty())
                             ? _device_data->maxspecial
                             : 0;
  const int* d_nspecial =
      (!_device_data->d_nspecial.empty())
          ? thrust::raw_pointer_cast(_device_data->d_nspecial.data())
          : nullptr;
  const rbmd::Id* d_special =
      (!_device_data->d_special.empty())
          ? thrust::raw_pointer_cast(_device_data->d_special.data())
          : nullptr;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  const auto start_vl_pair_kernel = StartDetailedDeviceTiming();
  op::SpecialLJCutCoulForceOp<device::DEVICE_GPU>()(
                  force_box,_cut_off, num_atoms,_alpha,_qqr2e,
                  thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
                  thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
                  _lj_num_types,
                  thrust::raw_pointer_cast(_d_lj_force12.data()),
                  thrust::raw_pointer_cast(_d_lj_force6.data()),
                  thrust::raw_pointer_cast(_list->_start_idx.data()),
                  thrust::raw_pointer_cast(_list->_end_idx.data()),
                  thrust::raw_pointer_cast(_list->_d_neighbors.data()),
                  d_nspecial,
                  d_special,
                  maxspecial,
                  sbw.w12,
                  sbw.w13,
                  sbw.w14,
                  thrust::raw_pointer_cast(_device_data->_d_charge.data()),
                  thrust::raw_pointer_cast(_device_data->_d_px.data()),
                  thrust::raw_pointer_cast(_device_data->_d_py.data()),
                  thrust::raw_pointer_cast(_device_data->_d_pz.data()),
                  thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
                  thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
                  thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()),
                  thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
                  thrust::raw_pointer_cast(_d_total_evdwl.data()),
                  thrust::raw_pointer_cast(_d_total_ecoul.data()));
  RecordDetailedDeviceTiming("VL-PairKernel", start_vl_pair_kernel);

  RecordDeviceTiming("Short-Range", start_verlet_force);
  DebugSyncForceStage("verlet_short_range_pre_d2h");
  // D2H
  thrust::host_vector<rbmd::Real> h_total_evdwl(_d_total_evdwl);
  thrust::host_vector<rbmd::Real> h_total_ecoul(_d_total_ecoul);
  _e_vdwl = GetGlobalRealSum(h_total_evdwl[0]);
  _e_coul = GetGlobalRealSum(h_total_ecoul[0]);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
    thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
    thrust::raw_pointer_cast(_device_data->_d_virial_lj.data()));

}

void CVFF::ComputeLJSOG() {
    //neighbor_list_build
  auto start = std::chrono::high_resolution_clock::now();
  _list = _neighbor_list_builder->Build();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<rbmd::Real> duration = end - start;

  TimingStatistics::Instance().record("Neighbor-List",duration.count());

  //
  auto start_verlet_force = std::chrono::high_resolution_clock::now();
  //
  thrust::device_vector<rbmd::Real> _d_total_evdwl(1, 0.0);
  thrust::device_vector<rbmd::Real> _d_total_ecoul(1, 0.0);


  //
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const auto sbw = read_special_bond_weights();
  const int maxspecial = (_device_data && !_device_data->d_nspecial.empty() &&
                          !_device_data->d_special.empty())
                             ? _device_data->maxspecial
                             : 0;
  const int* d_nspecial =
      (!_device_data->d_nspecial.empty())
          ? thrust::raw_pointer_cast(_device_data->d_nspecial.data())
          : nullptr;
  const rbmd::Id* d_special =
      (!_device_data->d_special.empty())
          ? thrust::raw_pointer_cast(_device_data->d_special.data())
          : nullptr;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::ComputeSpecialLJCutCoulForceUserOp<device::DEVICE_GPU>()(
                  force_box, _cut_off, num_atoms,_qqr2e,
                  _rbsog_sigma,_rbsog_b,_rbsog_Mmax,_w0,
                  thrust::raw_pointer_cast(_d_taylor_coeff.data()),
                  thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
                  thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
                  thrust::raw_pointer_cast(_device_data->_d_sigma.data()),
                  thrust::raw_pointer_cast(_device_data->_d_eps.data()),
                  thrust::raw_pointer_cast(_list->_start_idx.data()),
                  thrust::raw_pointer_cast(_list->_end_idx.data()),
                  thrust::raw_pointer_cast(_list->_d_neighbors.data()),
                  d_nspecial,
                  d_special,
                  maxspecial,
                  sbw.w12,
                  sbw.w13,
                  sbw.w14,
                  thrust::raw_pointer_cast(_device_data->_d_charge.data()),
                  thrust::raw_pointer_cast(_device_data->_d_px.data()),
                  thrust::raw_pointer_cast(_device_data->_d_py.data()),
                  thrust::raw_pointer_cast(_device_data->_d_pz.data()),
                  thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
                  thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
                  thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()),
                  thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
                  thrust::raw_pointer_cast(_d_total_evdwl.data()),
                  thrust::raw_pointer_cast(_d_total_ecoul.data()));

  auto end_verlet_force = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration_verlet_force = end_verlet_force - start_verlet_force;
  TimingStatistics::Instance().record("Short-Range",duration_verlet_force.count());
  DebugSyncForceStage("sog_short_range_pre_d2h");
  // D2H
  thrust::host_vector<rbmd::Real> h_total_evdwl(_d_total_evdwl);
  thrust::host_vector<rbmd::Real> h_total_ecoul(_d_total_ecoul);
  _e_vdwl = GetGlobalRealSum(h_total_evdwl[0]);
  _e_coul = GetGlobalRealSum(h_total_ecoul[0]);

  // thrust::host_vector<rbmd::Real> h_ljcoul_x = _device_data->_d_force_ljcoul_x;
  // thrust::host_vector<rbmd::Real> h_ljcoul_y = _device_data->_d_force_ljcoul_y;
  // thrust::host_vector<rbmd::Real> h_ljcoul_z = _device_data->_d_force_ljcoul_z;
  //
  // std::ofstream ljcoul_file("ljcoul_sog.txt");
  // auto atom_id_to_idx =
  //   LinkedCellLocator::GetInstance().GetLinkedCell()->_atom_id_to_idx;
  // if (ljcoul_file.is_open()) {
  //   for (rbmd::Id i = 0; i < h_ljcoul_x.size(); ++i) {
  //     auto index = atom_id_to_idx[i];
  //     ljcoul_file << i  << " " <<h_ljcoul_x[index] <<" " <<h_ljcoul_y[index] <<" " <<
  //     h_ljcoul_z[index]<< "\n";
  //   }
  //   ljcoul_file.close();
  // }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
    thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
    thrust::raw_pointer_cast(_device_data->_d_virial_lj.data()));
  // ReduceVirial(num_atoms,_device_data->_d_flat_virial_lj,_device_data->_d_virial_lj);

  if (test_current_step ==0 ) {
    thrust::host_vector<rbmd::Real> h_lj_virial =_device_data->_d_virial_lj;
    std::ofstream lj_file;
    if (OpenOutputFile(lj_file, "lj_sog_virial.txt", std::ios::out | std::ios::trunc)) {
      for (rbmd::Id i = 0; i < h_lj_virial.size(); ++i) {
        lj_file << i  << " " <<h_lj_virial[i]  << "\n";
      }
      lj_file.close();
    }

  }

}

void CVFF::ComputeLJCoulEnergy()
{
  // energy
  //neighbor_list_build
  //auto start = std::chrono::high_resolution_clock::now();
  _list = _neighbor_list_builder->Build();

  //auto end = std::chrono::high_resolution_clock::now();

  //std::chrono::duration<rbmd::Real> duration = end - start;
  //TimingStatistics::Instance().record("Neighbor-List",duration.count());


  thrust::device_vector<rbmd::Real> _d_total_evdwl(1, 0.0);
  thrust::device_vector<rbmd::Real> _d_total_ecoul(1, 0.0);

  auto num_atoms = *(_structure_info_data->_num_atoms);
  const auto sbw = read_special_bond_weights();
  const int maxspecial = (_device_data && !_device_data->d_nspecial.empty() &&
                          !_device_data->d_special.empty())
                             ? _device_data->maxspecial
                             : 0;
  const int* d_nspecial =
      (!_device_data->d_nspecial.empty())
          ? thrust::raw_pointer_cast(_device_data->d_nspecial.data())
          : nullptr;
  const rbmd::Id* d_special =
      (!_device_data->d_special.empty())
          ? thrust::raw_pointer_cast(_device_data->d_special.data())
          : nullptr;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::SpeciaLJCutCoulEnergyOp<device::DEVICE_GPU>()(
                force_box,_cut_off, num_atoms,_alpha,_qqr2e,
                thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
                thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
                thrust::raw_pointer_cast(_device_data->_d_sigma.data()),
                thrust::raw_pointer_cast(_device_data->_d_eps.data()),
                thrust::raw_pointer_cast(_list->_start_idx.data()),
                thrust::raw_pointer_cast(_list->_end_idx.data()),
                thrust::raw_pointer_cast(_list->_d_neighbors.data()),
                d_nspecial,
                d_special,
                maxspecial,
                sbw.w12,
                sbw.w13,
                sbw.w14,
                thrust::raw_pointer_cast(_device_data->_d_charge.data()),
                thrust::raw_pointer_cast(_device_data->_d_px.data()),
                thrust::raw_pointer_cast(_device_data->_d_py.data()),
                thrust::raw_pointer_cast(_device_data->_d_pz.data()),
                thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
                thrust::raw_pointer_cast(_d_total_evdwl.data()),
           thrust::raw_pointer_cast(_d_total_ecoul.data()));

  // D2H
  thrust::host_vector<rbmd::Real> h_total_evdwl(_d_total_evdwl);
  thrust::host_vector<rbmd::Real> h_total_ecoul(_d_total_ecoul);
  _e_vdwl = GetGlobalRealSum(h_total_evdwl[0]);
  _e_coul = GetGlobalRealSum(h_total_ecoul[0]);

  //sum virial on host
  ReduceVirial(num_atoms,_device_data->_d_flat_virial_lj,
_device_data->_d_virial_lj);
}

void CVFF::SumForces()
{
  TransformForces(_device_data->_d_fx,_device_data->_d_force_ljcoul_x,
    _device_data->_d_force_kspace_x,_device_data->_d_force_bond_x,
    _device_data->_d_force_angle_x,_device_data->_d_force_dihedral_x,
    _device_data->_d_force_improper_x);

  TransformForces(_device_data->_d_fy,_device_data->_d_force_ljcoul_y,
    _device_data->_d_force_kspace_y,_device_data->_d_force_bond_y,
    _device_data->_d_force_angle_y,_device_data->_d_force_dihedral_y,
    _device_data->_d_force_improper_y);

  TransformForces(_device_data->_d_fz,_device_data->_d_force_ljcoul_z,
    _device_data->_d_force_kspace_z,_device_data->_d_force_bond_z,
    _device_data->_d_force_angle_z,_device_data->_d_force_dihedral_z,
    _device_data->_d_force_improper_z);
}

void CVFF::ComputeBondForce()
{
  const auto start = StartDeviceTiming();

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto _atom_id_to_idx =
    linked_cell->_atom_id_to_idx;

  thrust::fill(_device_data->_d_force_bond_x.begin(),
    _device_data->_d_force_bond_x.end(), 0.0f);
  thrust::fill(_device_data->_d_force_bond_y.begin(),
    _device_data->_d_force_bond_y.end(), 0.0f);
  thrust::fill(_device_data->_d_force_bond_z.begin(),
    _device_data->_d_force_bond_z.end(), 0.0f);

  thrust::fill(_device_data->_d_flat_virial_bond_atom.begin(),
    _device_data->_d_flat_virial_bond_atom.end(), 0.0f);

  thrust::device_vector<rbmd::Real> d_total_ebond(1, 0.0);
  thrust::device_vector<unsigned long long> d_total_cbond(1, 0ULL);

  auto num_bonds = *(_structure_info_data->_num_bonds);
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const rbmd::Id atom_id_to_idx_size =
      static_cast<rbmd::Id>(_atom_id_to_idx.size());
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomBondPath()) {
    op::ComputeBondForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(_atom_id_to_idx.data()),
      atom_id_to_idx_size,
      thrust::raw_pointer_cast(_device_data->_d_bond_coeffs_k.data()),
      static_cast<int>(_device_data->_d_bond_coeffs_k.size()),
      thrust::raw_pointer_cast(_device_data->_d_bond_coeffs_equilibrium.data()),
      thrust::raw_pointer_cast(_device_data->d_num_bond.data()),
      thrust::raw_pointer_cast(_device_data->d_bond_type.data()),
      thrust::raw_pointer_cast(_device_data->d_bond_atom.data()),
      _device_data->bond_per_atom,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_z.data()),
      thrust::raw_pointer_cast(d_total_ebond.data()),
      thrust::raw_pointer_cast(d_total_cbond.data()));
  } else {
    op::ComputeBondForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_bonds,thrust::raw_pointer_cast(_atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_bond_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_bond_coeffs_equilibrium.data()),
      thrust::raw_pointer_cast(_device_data->_d_bond_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_bond_id0.data()),
      thrust::raw_pointer_cast(_device_data->_d_bond_id1.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_bond_atom.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_bond_list.data()),
      thrust::raw_pointer_cast(d_total_ebond.data()));
  }

  RecordDeviceTiming("Bond", start);
  DebugSyncForceStage("bond_pre_d2h");

  // D2H
  thrust::host_vector<rbmd::Real> h_total_ebond(d_total_ebond);
  _e_bond = GetGlobalRealSum(h_total_ebond[0]);

  ThermoStats::Instance().AddThermoData("bond",_e_bond);

  // //sum virial_bond  on host
//   ReduceVirial(num_atoms,_device_data->_d_flat_virial_bond_atom,
// _device_data->_d_virial_bond);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
  thrust::raw_pointer_cast(_device_data->_d_flat_virial_bond_atom.data()),
  thrust::raw_pointer_cast(_device_data->_d_virial_bond.data()));
}

void CVFF::ComputeAngleForce()
{
  const auto start = StartDeviceTiming();
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto atom_id_to_idx = linked_cell->_atom_id_to_idx;

  thrust::fill(_device_data->_d_force_angle_x.begin(),
  _device_data->_d_force_angle_x.end(), 0.0f);
  thrust::fill(_device_data->_d_force_angle_y.begin(),
    _device_data->_d_force_angle_y.end(), 0.0f);
  thrust::fill(_device_data->_d_force_angle_z.begin(),
    _device_data->_d_force_angle_z.end(), 0.0f);

  thrust::fill(_device_data->_d_flat_virial_angle_atom.begin(),
  _device_data->_d_flat_virial_angle_atom.end(), 0.0f);

  thrust::device_vector<rbmd::Real> d_total_eangle(1, 0.0);
  thrust::device_vector<unsigned long long> d_total_cangle(1, 0ULL);

  auto num_angles = *(_structure_info_data->_num_angles);
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomAnglePath()) {
    op::ComputeAngleForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_coeffs_equilibrium.data()),
      thrust::raw_pointer_cast(_device_data->d_num_angle.data()),
      thrust::raw_pointer_cast(_device_data->d_angle_type.data()),
      thrust::raw_pointer_cast(_device_data->d_angle_atom1.data()),
      thrust::raw_pointer_cast(_device_data->d_angle_atom2.data()),
      thrust::raw_pointer_cast(_device_data->d_angle_atom3.data()),
      _device_data->angle_per_atom,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_z.data()),
      thrust::raw_pointer_cast(d_total_eangle.data()),
      thrust::raw_pointer_cast(d_total_cangle.data()));
  } else {
    op::ComputeAngleForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_angles,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_coeffs_equilibrium.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_id0.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_id1.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_id2.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_angle_atom.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_angle_list.data()),
      thrust::raw_pointer_cast(d_total_eangle.data()));
  }

  RecordDeviceTiming("Angle", start);
  DebugSyncForceStage("angle_pre_d2h");
  // D2H
  thrust::host_vector<rbmd::Real> h_total_eangle(d_total_eangle);
  _e_angle = GetGlobalRealSum(h_total_eangle[0]);

  ThermoStats::Instance().AddThermoData("angle",_e_angle);

  //sum virial_angle on host
//   ReduceVirial(num_atoms,_device_data->_d_flat_virial_angle_atom,
// _device_data->_d_virial_angle);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_angle_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_angle.data()));
}

void CVFF::ComputeDihedralForce()
{
  auto dihedral_type = DataManager::getInstance().getConfigData()->Get
  <std::string>("dihedral_type", "hyper_parameters", "force_field");

  if (dihedral_type == "harmonic") {
    DihedralHarmonic();
  }
  else if (dihedral_type == "opls") {
    DihedralOPLS();
  }
  else if (dihedral_type == "fourier") {
    DihedralFourier();
  }
  //add thermo
  ThermoStats::Instance().AddThermoData("dihedral",_e_dihedral);
}

void CVFF::DihedralHarmonic() {
  const auto start = StartDeviceTiming();
  thrust::fill(_device_data->_d_force_dihedral_x.begin(),
    _device_data->_d_force_dihedral_x.end(), 0.0f);
  thrust::fill(_device_data->_d_force_dihedral_y.begin(),
    _device_data->_d_force_dihedral_y.end(), 0.0f);
  thrust::fill(_device_data->_d_force_dihedral_z.begin(),
    _device_data->_d_force_dihedral_z.end(), 0.0f);

  thrust::fill(_device_data->_d_flat_virial_dihedral_atom.begin(),
  _device_data->_d_flat_virial_dihedral_atom.end(), 0.0f);

  //thrust::device_vector<int4> dihedral_list;
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto atom_id_to_idx = linked_cell->_atom_id_to_idx;

  thrust::device_vector<rbmd::Real> d_total_edihedral(1, 0.0);
  thrust::device_vector<unsigned long long> d_total_cdihedral(1, 0ULL);

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_dihedrals = *(_structure_info_data->_num_dihedrals);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomDihedralPath()) {
    op::ComputeDihedralForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_sign.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_multiplicity.data()),
      thrust::raw_pointer_cast(_device_data->d_num_dihedral.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_type.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom1.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom2.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom3.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom4.data()),
      _device_data->dihedral_per_atom,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
      thrust::raw_pointer_cast(d_total_edihedral.data()),
      thrust::raw_pointer_cast(d_total_cdihedral.data()));
  } else {
    op::ComputeDihedralForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_dihedrals,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_sign.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_multiplicity.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id0.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id1.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id2.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id3.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_list.data()),
      thrust::raw_pointer_cast(d_total_edihedral.data()));
  }

  RecordDeviceTiming("Dihedral", start);

  // D2H
  thrust::host_vector<rbmd::Real> h_total_edihedral(d_total_edihedral);
  _e_dihedral = GetGlobalRealSum(h_total_edihedral[0]);

  //sum virial_dihedral on host
  // ReduceVirial(num_atoms,_device_data->_d_flat_virial_dihedral_atom,
  // _device_data->_d_virial_dihedral);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_dihedral.data()));
}

void CVFF::DihedralOPLS() {
   auto start = std::chrono::high_resolution_clock::now();
  thrust::fill(_device_data->_d_force_dihedral_x.begin(),
    _device_data->_d_force_dihedral_x.end(), 0.0f);
  thrust::fill(_device_data->_d_force_dihedral_y.begin(),
    _device_data->_d_force_dihedral_y.end(), 0.0f);
  thrust::fill(_device_data->_d_force_dihedral_z.begin(),
    _device_data->_d_force_dihedral_z.end(), 0.0f);

  thrust::fill(_device_data->_d_flat_virial_dihedral_atom.begin(),
  _device_data->_d_flat_virial_dihedral_atom.end(), 0.0f);

  //thrust::device_vector<int4> dihedral_list;
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto atom_id_to_idx = linked_cell->_atom_id_to_idx;

  thrust::device_vector<rbmd::Real> d_total_edihedral(1, 0.0);
  thrust::device_vector<unsigned long long> d_total_cdihedral(1, 0ULL);

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_dihedrals = *(_structure_info_data->_num_dihedrals);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomDihedralPath()) {
    op::ComputeDihedralOPLSForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k1.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k2.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k3.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k4.data()),
      thrust::raw_pointer_cast(_device_data->d_num_dihedral.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_type.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom1.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom2.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom3.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom4.data()),
      _device_data->dihedral_per_atom,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
      thrust::raw_pointer_cast(d_total_edihedral.data()),
      thrust::raw_pointer_cast(d_total_cdihedral.data()));
  } else {
    op::ComputeDihedralOPLSForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_dihedrals,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k1.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k2.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k3.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k4.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id0.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id1.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id2.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id3.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_list.data()),
      thrust::raw_pointer_cast(d_total_edihedral.data()));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record("Dihedral",duration.count());
  // D2H
  thrust::host_vector<rbmd::Real> h_total_edihedral(d_total_edihedral);
  _e_dihedral = GetGlobalRealSum(h_total_edihedral[0]);

  //sum virial_dihedral on host
  // ReduceVirial(num_atoms,_device_data->_d_flat_virial_dihedral_atom,
  //   _device_data->_d_virial_dihedral);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_dihedral.data()));
}

void CVFF::DihedralFourier() {
  std::cout<< "test--DihedralFourier"<<std::endl;
   auto start = std::chrono::high_resolution_clock::now();
  thrust::fill(_device_data->_d_force_dihedral_x.begin(),
    _device_data->_d_force_dihedral_x.end(), 0.0f);
  thrust::fill(_device_data->_d_force_dihedral_y.begin(),
    _device_data->_d_force_dihedral_y.end(), 0.0f);
  thrust::fill(_device_data->_d_force_dihedral_z.begin(),
    _device_data->_d_force_dihedral_z.end(), 0.0f);

  thrust::fill(_device_data->_d_flat_virial_dihedral_atom.begin(),
  _device_data->_d_flat_virial_dihedral_atom.end(), 0.0f);

  //thrust::device_vector<int4> dihedral_list;
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto atom_id_to_idx = linked_cell->_atom_id_to_idx;

  thrust::device_vector<rbmd::Real> d_total_edihedral(1, 0.0);
  thrust::device_vector<unsigned long long> d_total_cdihedral(1, 0ULL);

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_dihedrals = *(_structure_info_data->_num_dihedrals);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomDihedralPath()) {
    op::ComputeDihedralFourierForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_nterms.data()),
      thrust::raw_pointer_cast(_device_data->_d_fourier_offsets.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_multiplicity.data()),
      thrust::raw_pointer_cast(_device_data->_d_fourier_cos_shift.data()),
      thrust::raw_pointer_cast(_device_data->_d_fourier_sin_shift.data()),
      thrust::raw_pointer_cast(_device_data->d_num_dihedral.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_type.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom1.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom2.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom3.data()),
      thrust::raw_pointer_cast(_device_data->d_dihedral_atom4.data()),
      _device_data->dihedral_per_atom,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
      thrust::raw_pointer_cast(d_total_edihedral.data()),
      thrust::raw_pointer_cast(d_total_cdihedral.data()));
  } else {
    op::ComputeDihedralFourierForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_dihedrals,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_nterms.data()),
      thrust::raw_pointer_cast(_device_data->_d_fourier_offsets.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_coeffs_multiplicity.data()),
      thrust::raw_pointer_cast(_device_data->_d_fourier_cos_shift.data()),
      thrust::raw_pointer_cast(_device_data->_d_fourier_sin_shift.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id0.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id1.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id2.data()),
      thrust::raw_pointer_cast(_device_data->_d_dihedral_id3.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_list.data()),
      thrust::raw_pointer_cast(d_total_edihedral.data()));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record("Dihedral",duration.count());
  std::cout<< "test--end"<<std::endl;
  // D2H
  thrust::host_vector<rbmd::Real> h_total_edihedral(d_total_edihedral);
  _e_dihedral = GetGlobalRealSum(h_total_edihedral[0]);

  //sum virial_dihedral on host
  // ReduceVirial(num_atoms,_device_data->_d_flat_virial_dihedral_atom,
  // _device_data->_d_virial_dihedral);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_dihedral.data()));

}

void CVFF::ComputeImproperForce()
{
  auto improper_type = DataManager::getInstance().getConfigData()->Get
<std::string>("improper_type", "hyper_parameters", "force_field");

  if (improper_type == "harmonic") {
    ImproperHarmonic();
  }
  else if (improper_type == "cvff") {
    ImproperCVFF();
  }

  //add thermo
  ThermoStats::Instance().AddThermoData("improper",_e_improper);
}

void CVFF::ImproperHarmonic() {
  auto start = std::chrono::high_resolution_clock::now();
  thrust::fill(_device_data->_d_force_improper_x.begin(),
    _device_data->_d_force_improper_x.end(), 0.0f);
  thrust::fill(_device_data->_d_force_improper_y.begin(),
    _device_data->_d_force_improper_y.end(), 0.0f);
  thrust::fill(_device_data->_d_force_improper_z.begin(),
    _device_data->_d_force_improper_z.end(), 0.0f);

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto atom_id_to_idx = linked_cell->_atom_id_to_idx;

  thrust::device_vector<rbmd::Real> d_total_eimproper(1, 0.0);
  thrust::device_vector<unsigned long long> d_total_cimproper(1, 0ULL);

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_impropers = *(_structure_info_data->_num_impropers);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomImproperPath()) {
    op::ComputeImproperHarmonicPerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_chi.data()),
      thrust::raw_pointer_cast(_device_data->d_num_improper.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_type.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom1.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom2.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom3.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom4.data()),
      _device_data->improper_per_atom,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
      thrust::raw_pointer_cast(d_total_eimproper.data()),
      thrust::raw_pointer_cast(d_total_cimproper.data()));
  } else {
    op::ComputeImproperHarmonicForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_impropers,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_chi.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id0.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id1.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id2.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id3.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
      thrust::raw_pointer_cast(d_total_eimproper.data()));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record("Improper",duration.count());

  // D2H
  thrust::host_vector<rbmd::Real> h_total_eimproper(d_total_eimproper);
  _e_improper = GetGlobalRealSum(h_total_eimproper[0]);

  // ReduceVirial(num_atoms,_device_data->_d_flat_virial_improper_atom,
  // _device_data->_d_virial_improper);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_improper.data()));
}

void CVFF::ImproperCVFF()
{
   auto start = std::chrono::high_resolution_clock::now();
  thrust::fill(_device_data->_d_force_improper_x.begin(),
    _device_data->_d_force_improper_x.end(), 0.0f);
  thrust::fill(_device_data->_d_force_improper_y.begin(),
    _device_data->_d_force_improper_y.end(), 0.0f);
  thrust::fill(_device_data->_d_force_improper_z.begin(),
    _device_data->_d_force_improper_z.end(), 0.0f);

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto atom_id_to_idx = linked_cell->_atom_id_to_idx;

  thrust::device_vector<rbmd::Real> d_total_eimproper(1, 0.0);
  thrust::device_vector<unsigned long long> d_total_cimproper(1, 0ULL);

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_impropers = *(_structure_info_data->_num_impropers);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomImproperPath()) {
    op::ComputeImproperCVFFPerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_d.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_n.data()),
      thrust::raw_pointer_cast(_device_data->d_num_improper.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_type.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom1.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom2.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom3.data()),
      thrust::raw_pointer_cast(_device_data->d_improper_atom4.data()),
      _device_data->improper_per_atom,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
      thrust::raw_pointer_cast(d_total_eimproper.data()),
      thrust::raw_pointer_cast(d_total_cimproper.data()));
  } else {
    op::ComputeImproperCVFFForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_impropers,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_k.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_d.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_coeffs_n.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id0.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id1.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id2.data()),
      thrust::raw_pointer_cast(_device_data->_d_improper_id3.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
      thrust::raw_pointer_cast(d_total_eimproper.data()));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record("Improper",duration.count());

  // D2H
  thrust::host_vector<rbmd::Real> h_total_eimproper(d_total_eimproper);
  _e_improper = GetGlobalRealSum(h_total_eimproper[0]);

//   ReduceVirial(num_atoms,_device_data->_d_flat_virial_improper_atom,
// _device_data->_d_virial_improper);

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_improper.data()));
}

void CVFF::EvaluatePotentialEnergy()
{
  _e_kspace = _kspace_calculator->GetKspacEnergy();
  ThermoStats::Instance().AddThermoData("kspace",_e_kspace);

  _e_pe_rbl = _e_vdwl_rbl + _e_coul_rbl+_e_kspace+
                  _e_bond + _e_angle+_e_dihedral+_e_improper;

  _e_pe = _e_vdwl+ _e_coul +_e_kspace+
              _e_bond +_e_angle +_e_dihedral+_e_improper;

  ThermoStats::Instance().AddThermoData("total-potential-energy",_e_pe);

  //out
  auto interval = DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
"interval", "outputs", "thermo_out");

  std::ofstream outfile;
  if (!OpenOutputFile(outfile, "thermo.txt", std::ios::app)) {
    return;
  }

  if (outfile.tellp() == 0) {
    outfile << "step  e_vdwl  e_coul  e_kspace  e_bond "
            << "e_angle  e_dihedral e_improper e_pe" << std::endl;
  }
  if (test_current_step % interval == 0) {
    outfile << test_current_step << " " << _e_vdwl << " " << _e_coul << " "
        << _e_kspace << " " << _e_bond << " " << _e_angle << " "
        << _e_dihedral << " " << _e_improper << " " <<  _e_pe << std::endl;
  }
  outfile.close();
}
