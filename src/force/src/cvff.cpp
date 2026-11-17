#include "common/revision_timing.h"
#include "cvff.h"
#include "common/lj_mixing.h"
#include "force_box_selector.h"

//#include <output/include/Logger.hpp>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <limits>
#include <iomanip>
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
#include "thrust/fill.h"
#include "thrust/host_vector.h"
#include "thrust/sort.h"
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

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool EnvFlagEnabledOrDefault(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::string(value) != "0";
}

template <typename T>
void ClearDeviceVectorAsync(thrust::device_vector<T>& values,
                            STREAM_T stream) {
  if (values.empty()) {
    return;
  }
  CHECK_RUNTIME(MEMSET_ASYNC(thrust::raw_pointer_cast(values.data()), 0,
                             values.size() * sizeof(T), stream));
}

std::string NormalizeConfigToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::toupper(ch));
                 });
  return value;
}

int EnvIntOrDefault(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return fallback;
  }
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

rbmd::Id PairThreadsPerParticle(const char* override_env,
                                rbmd::Id estimated_pair_count) {
  const char* value = std::getenv(override_env);
  if (value != nullptr && value[0] != '\0') {
    const rbmd::Id requested = static_cast<rbmd::Id>(std::atoi(value));
    switch (requested) {
      case 1:
      case 4:
      case 8:
      case 16:
      case 32:
      case 64:
        return std::min<rbmd::Id>(requested, WARP_SIZE);
      default:
        break;
    }
  }

  // Force evaluation has substantially more arithmetic per neighbor than
  // list construction. Keep roughly sixteen pairs per lane, then round up to
  // a supported logical subgroup size.
  const rbmd::Id required_lanes = std::max<rbmd::Id>(
      (std::max<rbmd::Id>(estimated_pair_count, 1) + 15) / 16, 1);
  rbmd::Id threads_per_particle = 1;
  while (threads_per_particle < required_lanes &&
         threads_per_particle < WARP_SIZE) {
    threads_per_particle *= 2;
  }
  if (threads_per_particle == 2) {
    threads_per_particle = 4;
  }
  return std::min<rbmd::Id>(threads_per_particle, WARP_SIZE);
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
  if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();
  return DeviceTimingClock::now();
}

void RecordDeviceTiming(const char* category,
                        DeviceTimingClock::time_point start) {
  if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();
  const auto end = DeviceTimingClock::now();
  const std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record(category, duration.count());
}

bool DetailedPairTimingEnabled() {
  static const bool enabled = rbmd::revision::GetOptions().isolated_timing ||
      EnvFlagEnabled("RBMD_DETAILED_PAIR_TIMING");
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

CVFF::~CVFF() {
  // KSpace may need to drain a queued producer on our stream during unwinding.
  _kspace_calculator.reset();
  DestroyForceOverlapResources();
}

bool CVFF::ForceOverlapEnabled() const {
  return _force_overlap_enabled;
}

void CVFF::ConfigureForceOverlap() {
  _bonded_overlap_enabled =
      EnvFlagEnabledOrDefault("RBMD_BONDED_OVERLAP", true);
  const int local_enabled =
      !rbmd::revision::GetOptions().isolated_timing &&
      EnvFlagEnabledOrDefault("RBMD_FORCE_OVERLAP", true) &&
      _kspace_calculator && _kspace_calculator->SupportsRbeOverlap();
  int enabled = local_enabled;
#ifdef USE_MPI
  // The serial and overlap schedules issue collectives in different orders.
  // Resolve the choice once, on every rank, before either schedule can run.
  int any_enabled = local_enabled;
  MPI_CHECK(MPI_Allreduce(&local_enabled, &enabled, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allreduce(&local_enabled, &any_enabled, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD));
  if (enabled != any_enabled) {
    Logger::Instance().warn(
        "CVFF force-overlap settings differ across ranks; using the serial "
        "schedule on all ranks");
  }
#endif
  _force_overlap_enabled = enabled != 0;
}

bool CVFF::CanOverlapVerletRbe() const {
  return ForceOverlapEnabled() && _neighbor_type != "RBL" &&
         _neighbor_type != "VERLET-SOG";
}

bool CVFF::CanOverlapRblRbe() const {
  return ForceOverlapEnabled() && _neighbor_type == "RBL";
}

bool CVFF::RblEnergyEnabled() const {
  return _energy_rbl_flag == "yes";
}

void CVFF::InitializeForceOverlapResources() {
  if (_force_overlap_resources_ready ||
      (!CanOverlapVerletRbe() && !CanOverlapRblRbe())) {
    return;
  }

  bool pair_stream_created = false;
  bool kspace_stream_created = false;
  bool bonded_stream_created = false;
  bool angle_stream_created = false;
  bool dihedral_stream_created = false;
  bool improper_stream_created = false;
  bool join_stream_created = false;
  bool atoms_ready_created = false;
  bool rho_ready_created = false;
  bool rbl_correction_ready_created = false;
  bool pair_begin_created = false;
  bool pair_done_created = false;
  bool kspace_begin_created = false;
  bool kspace_done_created = false;
  bool bonded_begin_created = false;
  bool angle_done_created = false;
  bool dihedral_done_created = false;
  bool improper_done_created = false;
  bool bonded_done_created = false;
  bool forces_joined_created = false;

  pair_stream_created = CHECK_RUNTIME(
      STREAM_CREATE_WITH_FLAGS(&_pair_stream, STREAM_NON_BLOCKING));
  kspace_stream_created =
      pair_stream_created && CHECK_RUNTIME(STREAM_CREATE_WITH_FLAGS(
                                 &_kspace_stream, STREAM_NON_BLOCKING));
  bonded_stream_created =
      kspace_stream_created && CHECK_RUNTIME(STREAM_CREATE_WITH_FLAGS(
                                   &_bonded_stream, STREAM_NON_BLOCKING));
  angle_stream_created =
      bonded_stream_created && CHECK_RUNTIME(STREAM_CREATE_WITH_FLAGS(
                                    &_angle_stream, STREAM_NON_BLOCKING));
  dihedral_stream_created =
      angle_stream_created && CHECK_RUNTIME(STREAM_CREATE_WITH_FLAGS(
                                  &_dihedral_stream, STREAM_NON_BLOCKING));
  improper_stream_created =
      dihedral_stream_created && CHECK_RUNTIME(STREAM_CREATE_WITH_FLAGS(
                                     &_improper_stream, STREAM_NON_BLOCKING));
  join_stream_created =
      improper_stream_created && CHECK_RUNTIME(STREAM_CREATE_WITH_FLAGS(
                                      &_join_stream, STREAM_NON_BLOCKING));
  atoms_ready_created = join_stream_created && CHECK_RUNTIME(
      NEW_FLAG_EVENT(&_atoms_ready, EVENT_DISABLE));
  rho_ready_created = atoms_ready_created &&
                      CHECK_RUNTIME(NEW_FLAG_EVENT(&_rho_ready, EVENT_DISABLE));
  rbl_correction_ready_created =
      rho_ready_created && CHECK_RUNTIME(NEW_EVENT(&_rbl_correction_ready));
  pair_begin_created =
      rbl_correction_ready_created && CHECK_RUNTIME(NEW_EVENT(&_pair_begin));
  pair_done_created =
      pair_begin_created && CHECK_RUNTIME(NEW_EVENT(&_pair_done));
  kspace_begin_created =
      pair_done_created && CHECK_RUNTIME(NEW_EVENT(&_kspace_begin));
  kspace_done_created =
      kspace_begin_created && CHECK_RUNTIME(NEW_EVENT(&_kspace_done));
  bonded_begin_created =
      kspace_done_created && CHECK_RUNTIME(NEW_EVENT(&_bonded_begin));
  angle_done_created =
      bonded_begin_created &&
      CHECK_RUNTIME(NEW_FLAG_EVENT(&_angle_done, EVENT_DISABLE));
  dihedral_done_created =
      angle_done_created &&
      CHECK_RUNTIME(NEW_FLAG_EVENT(&_dihedral_done, EVENT_DISABLE));
  improper_done_created =
      dihedral_done_created &&
      CHECK_RUNTIME(NEW_FLAG_EVENT(&_improper_done, EVENT_DISABLE));
  bonded_done_created =
      improper_done_created && CHECK_RUNTIME(NEW_EVENT(&_bonded_done));
  forces_joined_created =
      bonded_done_created &&
      CHECK_RUNTIME(NEW_FLAG_EVENT(&_forces_joined, EVENT_DISABLE));

  const bool buffers_ready = forces_joined_created && CHECK_RUNTIME(
      MALLOCHOST(reinterpret_cast<void**>(&_h_overlap_values),
                 9 * sizeof(rbmd::Real)));
  if (!buffers_ready) {
    if (forces_joined_created) { EVENT_DESTROY(_forces_joined); }
    if (_h_overlap_values) {
      FREE_PINNED_HOST(_h_overlap_values);
      _h_overlap_values = nullptr;
    }
    if (bonded_done_created) {
      EVENT_DESTROY(_bonded_done);
    }
    if (improper_done_created) {
      EVENT_DESTROY(_improper_done);
    }
    if (dihedral_done_created) {
      EVENT_DESTROY(_dihedral_done);
    }
    if (angle_done_created) {
      EVENT_DESTROY(_angle_done);
    }
    if (bonded_begin_created) {
      EVENT_DESTROY(_bonded_begin);
    }
    if (kspace_done_created) {
      EVENT_DESTROY(_kspace_done);
    }
    if (kspace_begin_created) {
      EVENT_DESTROY(_kspace_begin);
    }
    if (pair_done_created) {
      EVENT_DESTROY(_pair_done);
    }
    if (pair_begin_created) {
      EVENT_DESTROY(_pair_begin);
    }
    if (rbl_correction_ready_created) {
      EVENT_DESTROY(_rbl_correction_ready);
    }
    if (rho_ready_created) {
      EVENT_DESTROY(_rho_ready);
    }
    if (atoms_ready_created) {
      EVENT_DESTROY(_atoms_ready);
    }
    if (join_stream_created) {
      STREAM_DESTROY(_join_stream);
    }
    if (improper_stream_created) {
      STREAM_DESTROY(_improper_stream);
    }
    if (dihedral_stream_created) {
      STREAM_DESTROY(_dihedral_stream);
    }
    if (angle_stream_created) {
      STREAM_DESTROY(_angle_stream);
    }
    if (bonded_stream_created) {
      STREAM_DESTROY(_bonded_stream);
    }
    if (kspace_stream_created) {
      STREAM_DESTROY(_kspace_stream);
    }
    if (pair_stream_created) {
      STREAM_DESTROY(_pair_stream);
    }
    throw std::runtime_error("Failed to create CVFF force-overlap resources");
  }
  _force_overlap_resources_ready = true;
  Logger::Instance().info(
      "CVFF force overlap enabled: pair and bonded work overlap RBE rho/MPI; "
      "set RBMD_FORCE_OVERLAP=0 for the serial fallback");
}

void CVFF::DestroyForceOverlapResources() {
  if (!_force_overlap_resources_ready) {
    return;
  }

#ifdef USE_MPI
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    if (_correction_request != MPI_REQUEST_NULL) {
      MPI_Wait(&_correction_request, MPI_STATUS_IGNORE);
    }
    if (_energy_request != MPI_REQUEST_NULL) {
      MPI_Wait(&_energy_request, MPI_STATUS_IGNORE);
    }
  }
#endif
  STREAM_SYNC(_pair_stream);
  STREAM_SYNC(_kspace_stream);
  STREAM_SYNC(_bonded_stream);
  STREAM_SYNC(_angle_stream);
  STREAM_SYNC(_dihedral_stream);
  STREAM_SYNC(_improper_stream);
  STREAM_SYNC(_join_stream);
  EVENT_DESTROY(_atoms_ready);
  EVENT_DESTROY(_rho_ready);
  EVENT_DESTROY(_rbl_correction_ready);
  EVENT_DESTROY(_pair_begin);
  EVENT_DESTROY(_pair_done);
  EVENT_DESTROY(_kspace_begin);
  EVENT_DESTROY(_kspace_done);
  EVENT_DESTROY(_bonded_begin);
  EVENT_DESTROY(_angle_done);
  EVENT_DESTROY(_dihedral_done);
  EVENT_DESTROY(_improper_done);
  EVENT_DESTROY(_bonded_done);
  EVENT_DESTROY(_forces_joined);
  STREAM_DESTROY(_pair_stream);
  STREAM_DESTROY(_kspace_stream);
  STREAM_DESTROY(_bonded_stream);
  STREAM_DESTROY(_angle_stream);
  STREAM_DESTROY(_dihedral_stream);
  STREAM_DESTROY(_improper_stream);
  STREAM_DESTROY(_join_stream);
  FREE_PINNED_HOST(_h_overlap_values);
  _h_overlap_values = nullptr;
  _force_overlap_resources_ready = false;
}

void CVFF::RecordOverlapTiming(const char* category, EVENT_T begin,
                               EVENT_T end) const {
  rbmd::revision::RequireGpu(EVENT_SYNC(end));
  float elapsed_ms = 0.0f;
  if (CHECK_RUNTIME(EVENT_ELAPSED_TIME(&elapsed_ms, begin, end))) {
    TimingStatistics::Instance().record(
        category, static_cast<rbmd::Real>(elapsed_ms * 1.0e-3f));
  }
}

void CVFF::QueueOverlapPairEnergy() {
  CHECK_RUNTIME(MEMCPY_ASYNC(
      _h_overlap_values + 3, thrust::raw_pointer_cast(_d_total_evdwl.data()),
      sizeof(rbmd::Real), D2H, _pair_stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(
      _h_overlap_values + 4, thrust::raw_pointer_cast(_d_total_ecoul.data()),
      sizeof(rbmd::Real), D2H, _pair_stream));
}

void CVFF::LaunchOverlapBondedForces() {
  CHECK_RUNTIME(EVENT_RECORD(_bonded_begin, _bonded_stream));
  _h_overlap_values[7] = rbmd::Real(0);
  _h_overlap_values[8] = rbmd::Real(0);

  if (!_bonded_overlap_enabled) {
    ComputeBondForce(_bonded_stream, true);
    ComputeAngleForce(_bonded_stream, true);
    if (*(_structure_info_data->_num_dihedrals)) {
      ComputeDihedralForce(_bonded_stream, true);
    }
    if (*(_structure_info_data->_num_impropers)) {
      ComputeImproperForce(_bonded_stream, true);
    }
    CHECK_RUNTIME(MEMCPY_ASYNC(
        _h_overlap_values + 5, thrust::raw_pointer_cast(_d_total_ebond.data()),
        sizeof(rbmd::Real), D2H, _bonded_stream));
    CHECK_RUNTIME(MEMCPY_ASYNC(
        _h_overlap_values + 6, thrust::raw_pointer_cast(_d_total_eangle.data()),
        sizeof(rbmd::Real), D2H, _bonded_stream));
    if (*(_structure_info_data->_num_dihedrals)) {
      CHECK_RUNTIME(MEMCPY_ASYNC(
          _h_overlap_values + 7,
          thrust::raw_pointer_cast(_d_total_edihedral.data()),
          sizeof(rbmd::Real), D2H, _bonded_stream));
    }
    if (*(_structure_info_data->_num_impropers)) {
      CHECK_RUNTIME(MEMCPY_ASYNC(
          _h_overlap_values + 8,
          thrust::raw_pointer_cast(_d_total_eimproper.data()),
          sizeof(rbmd::Real), D2H, _bonded_stream));
    }
    CHECK_RUNTIME(EVENT_RECORD(_bonded_done, _bonded_stream));
    return;
  }

  CHECK_RUNTIME(STREAM_WAIT_EVENT(_angle_stream, _bonded_begin));
  CHECK_RUNTIME(STREAM_WAIT_EVENT(_dihedral_stream, _bonded_begin));
  CHECK_RUNTIME(STREAM_WAIT_EVENT(_improper_stream, _bonded_begin));

  ComputeBondForce(_bonded_stream, true);
  CHECK_RUNTIME(MEMCPY_ASYNC(
      _h_overlap_values + 5, thrust::raw_pointer_cast(_d_total_ebond.data()),
      sizeof(rbmd::Real), D2H, _bonded_stream));

  ComputeAngleForce(_angle_stream, true);
  CHECK_RUNTIME(MEMCPY_ASYNC(
      _h_overlap_values + 6, thrust::raw_pointer_cast(_d_total_eangle.data()),
      sizeof(rbmd::Real), D2H, _angle_stream));
  CHECK_RUNTIME(EVENT_RECORD(_angle_done, _angle_stream));

  if (*(_structure_info_data->_num_dihedrals)) {
    ComputeDihedralForce(_dihedral_stream, true);
    CHECK_RUNTIME(MEMCPY_ASYNC(
        _h_overlap_values + 7,
        thrust::raw_pointer_cast(_d_total_edihedral.data()),
        sizeof(rbmd::Real), D2H, _dihedral_stream));
    CHECK_RUNTIME(EVENT_RECORD(_dihedral_done, _dihedral_stream));
  }
  if (*(_structure_info_data->_num_impropers)) {
    ComputeImproperForce(_improper_stream, true);
    CHECK_RUNTIME(MEMCPY_ASYNC(
        _h_overlap_values + 8,
        thrust::raw_pointer_cast(_d_total_eimproper.data()),
        sizeof(rbmd::Real), D2H, _improper_stream));
    CHECK_RUNTIME(EVENT_RECORD(_improper_done, _improper_stream));
  }

  CHECK_RUNTIME(STREAM_WAIT_EVENT(_bonded_stream, _angle_done));
  if (*(_structure_info_data->_num_dihedrals)) {
    CHECK_RUNTIME(STREAM_WAIT_EVENT(_bonded_stream, _dihedral_done));
  }
  if (*(_structure_info_data->_num_impropers)) {
    CHECK_RUNTIME(STREAM_WAIT_EVENT(_bonded_stream, _improper_done));
  }
  CHECK_RUNTIME(EVENT_RECORD(_bonded_done, _bonded_stream));
}

void CVFF::StartOverlapRblCorrection() {
  CHECK_RUNTIME(EVENT_SYNC(_rbl_correction_ready));
#ifdef USE_MPI
  MPI_CHECK(MPI_Iallreduce(MPI_IN_PLACE, _h_overlap_values, 3, MPI_RBMD_REAL,
                          MPI_SUM, MPI_COMM_WORLD, &_correction_request));
#endif
}

void CVFF::FinishOverlapRblCorrection() {
  const auto start = DeviceTimingClock::now();
#ifdef USE_MPI
  MPI_CHECK(MPI_Wait(&_correction_request, MPI_STATUS_IGNORE));
#endif
  const std::chrono::duration<double> wait = DeviceTimingClock::now() - start;
  TimingStatistics::Instance().record("RBL-CorrectionMPIWait", wait.count());
  _corr_value_x = _h_overlap_values[0] / _global_num_atoms;
  _corr_value_y = _h_overlap_values[1] / _global_num_atoms;
  _corr_value_z = _h_overlap_values[2] / _global_num_atoms;
}

void CVFF::StartOverlapEnergyReduction() {
  CHECK_RUNTIME(EVENT_SYNC(_pair_done));
  CHECK_RUNTIME(EVENT_SYNC(_bonded_done));
#ifdef USE_MPI
  MPI_CHECK(MPI_Iallreduce(MPI_IN_PLACE, _h_overlap_values + 3, 6,
                          MPI_RBMD_REAL, MPI_SUM, MPI_COMM_WORLD,
                          &_energy_request));
#endif
}

void CVFF::FinishOverlapEnergyReduction() {
  const auto start = DeviceTimingClock::now();
#ifdef USE_MPI
  MPI_CHECK(MPI_Wait(&_energy_request, MPI_STATUS_IGNORE));
#endif
  const std::chrono::duration<double> wait = DeviceTimingClock::now() - start;
  TimingStatistics::Instance().record("Force-EnergyMPIWait", wait.count());
  _e_vdwl = _h_overlap_values[3];
  _e_coul = _h_overlap_values[4];
  _e_bond = _h_overlap_values[5];
  _e_angle = _h_overlap_values[6];
  _e_dihedral = _h_overlap_values[7];
  _e_improper = _h_overlap_values[8];
  auto& thermo = ThermoStats::Instance();
  thermo.AddThermoData("vdwl", _e_vdwl);
  thermo.AddThermoData("coul", _e_coul);
  thermo.AddThermoData("bond", _e_bond);
  thermo.AddThermoData("angle", _e_angle);
  if (*(_structure_info_data->_num_dihedrals)) {
    thermo.AddThermoData("dihedral", _e_dihedral);
  }
  if (*(_structure_info_data->_num_impropers)) {
    thermo.AddThermoData("improper", _e_improper);
  }
}

void CVFF::InitializeRblCoulombLookup() {
  constexpr rbmd::Id kDefaultIntervals = 1024;
  constexpr double kCoreRadius = 2.0;
  constexpr double kAllowedRelativeError = 5.0e-6;
  const double alpha = static_cast<double>(_alpha);
  const double cutoff = static_cast<double>(_cut_off);
  if (!(alpha > 0.0) || !(cutoff > 0.0)) {
    throw std::runtime_error("Invalid RBL Coulomb lookup parameters");
  }

  const double beta = std::sqrt(alpha);
  const double sqrt_pi = std::sqrt(M_PI);
  const double coul_exp_prefactor = 2.0 * beta / sqrt_pi;
  const auto screened_coefficient = [&](double distance_2) {
    const double distance = std::sqrt(distance_2);
    return std::erfc(beta * distance) /
               (distance_2 * distance) +
           coul_exp_prefactor * std::exp(-alpha * distance_2) / distance_2;
  };
  const auto screened_derivative = [&](double distance_2) {
    const double distance = std::sqrt(distance_2);
    const double erfc_value = std::erfc(beta * distance);
    const double exp_value = std::exp(-alpha * distance_2);
    return -beta / sqrt_pi * exp_value /
               (distance_2 * distance_2) -
           1.5 * erfc_value /
               (distance_2 * distance_2 * distance) +
           coul_exp_prefactor * exp_value *
               (-alpha / distance_2 -
                1.0 / (distance_2 * distance_2));
  };

  _rbl_table_r2_min = static_cast<rbmd::Real>(
      std::min(kCoreRadius * kCoreRadius, cutoff * cutoff * 0.25));
  _rbl_table_intervals = kDefaultIntervals;
  const double table_min = static_cast<double>(_rbl_table_r2_min);
  const double table_max = cutoff * cutoff;
  const double spacing =
      (table_max - table_min) / static_cast<double>(_rbl_table_intervals);
  if (!(spacing > 0.0)) {
    throw std::runtime_error("RBL Coulomb lookup interval is empty");
  }
  _rbl_table_scale = static_cast<rbmd::Real>(1.0 / spacing);

  thrust::host_vector<rbmd::Real> coefficients(
      static_cast<std::size_t>(_rbl_table_intervals) * 4);
  for (rbmd::Id interval = 0; interval < _rbl_table_intervals; ++interval) {
    const double x0 = table_min + spacing * interval;
    const double x1 = x0 + spacing;
    const double value0 = screened_coefficient(x0);
    const double value1 = screened_coefficient(x1);
    const double slope0 = spacing * screened_derivative(x0);
    const double slope1 = spacing * screened_derivative(x1);
    const std::size_t base = static_cast<std::size_t>(interval) * 4;
    coefficients[base + 0] = static_cast<rbmd::Real>(
        2.0 * value0 - 2.0 * value1 + slope0 + slope1);
    coefficients[base + 1] = static_cast<rbmd::Real>(
        -3.0 * value0 + 3.0 * value1 - 2.0 * slope0 - slope1);
    coefficients[base + 2] = static_cast<rbmd::Real>(slope0);
    coefficients[base + 3] = static_cast<rbmd::Real>(value0);
  }

  double factorial = 1.0;
  for (int n = 1; n <= 6; ++n) {
    factorial *= static_cast<double>(n);
    const double sign = (n % 2 == 0) ? 1.0 : -1.0;
    const double coefficient =
        4.0 * n * sign * std::pow(beta, 2 * n + 1) /
        (sqrt_pi * factorial * (2 * n + 1));
    _rbl_core_taylor[static_cast<std::size_t>(n - 1)] =
        static_cast<rbmd::Real>(coefficient);
  }

  double max_relative_error = 0.0;
  constexpr int kValidationSamples = 32768;
  for (int sample = 0; sample <= kValidationSamples; ++sample) {
    const double fraction =
        static_cast<double>(sample) / kValidationSamples;
    const double distance_2 =
        1.0e-4 + fraction * (table_max - 1.0e-4);
    const double exact = screened_coefficient(distance_2);
    double approximate = 0.0;
    if (distance_2 < table_min) {
      double polynomial = static_cast<double>(_rbl_core_taylor[5]);
      for (int index = 4; index >= 0; --index) {
        polynomial = polynomial * distance_2 +
                     static_cast<double>(_rbl_core_taylor[index]);
      }
      approximate = 1.0 /
                        (distance_2 * std::sqrt(distance_2)) +
                    polynomial;
    } else {
      const double position =
          (distance_2 - table_min) *
          static_cast<double>(_rbl_table_scale);
      const rbmd::Id interval = std::min<rbmd::Id>(
          static_cast<rbmd::Id>(position), _rbl_table_intervals - 1);
      const double local = position - interval;
      const std::size_t base = static_cast<std::size_t>(interval) * 4;
      approximate =
          ((static_cast<double>(coefficients[base + 0]) * local +
            static_cast<double>(coefficients[base + 1])) *
               local +
           static_cast<double>(coefficients[base + 2])) *
              local +
          static_cast<double>(coefficients[base + 3]);
    }
    max_relative_error = std::max(
        max_relative_error,
        std::abs(approximate - exact) / std::max(std::abs(exact), 1.0e-30));
  }
  if (max_relative_error > kAllowedRelativeError) {
    throw std::runtime_error("RBL Coulomb lookup accuracy guard failed");
  }
  _d_rbl_coulomb_coeff = coefficients;
  _d_rbl_core_taylor = thrust::host_vector<rbmd::Real>(
      _rbl_core_taylor.begin(), _rbl_core_taylor.end());
  Logger::Instance().info(
      "RBL Coulomb lookup: {} intervals, r2_min {}, max relative error {}",
      _rbl_table_intervals, _rbl_table_r2_min, max_relative_error);
}

void CVFF::Init()
{
  const auto& config = DataManager::getInstance().getConfigData();

  //neighbor
  _cut_off = config->Get<rbmd::Real>("cut_off", "hyper_parameters", "neighbor");
  _neighbor_type = NormalizeConfigToken(
      config->Get<std::string>("type", "hyper_parameters", "neighbor"));

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
  Logger::Instance().info("CVFF short-range dispatch: {}", _neighbor_type);
  if ("RBL" == _neighbor_type) {
    InitializeRblCoulombLookup();
  }

  const auto force_field_token = [&](const char* key) {
    return config->PathExists({"hyper_parameters", "force_field", key})
        ? NormalizeConfigToken(config->Get<std::string>(
              key, "hyper_parameters", "force_field"))
        : std::string{};
  };
  _geometric_lj_mixing = rbmd::UseGeometricLJMixing(
      force_field_token("dihedral_type"), force_field_token("mixing_rule"));
  Logger::Instance().info("LJ mixing rule: {}",
                         _geometric_lj_mixing ? "geometric" : "arithmetic");

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
          rbmd::MixLJSigma(h_sigma[type_i], h_sigma[type_j],
                          _geometric_lj_mixing);
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
  if ("RBL" == _neighbor_type) {
    _d_rbl_force_sum.resize(3);
  }
  _global_num_atoms = GetGlobalIdSum(*(_structure_info_data->_num_atoms));
  if (_global_num_atoms <= 0) {
    throw std::runtime_error("CVFF global atom count must be positive");
  }
  _d_total_evdwl.resize(1);
  _d_total_ecoul.resize(1);
  _d_total_ebond.resize(1);
  _d_total_eangle.resize(1);
  _d_total_edihedral.resize(1);
  _d_total_eimproper.resize(1);
  _d_total_cbond.resize(1);
  _d_total_cangle.resize(1);
  _d_total_cdihedral.resize(1);
  _d_total_cimproper.resize(1);
  ConfigureForceOverlap();
  InitializeForceOverlapResources();

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

  if (CanOverlapRblRbe()) {
    ExecuteRblRbeOverlap();
    return;
  }
  if (CanOverlapVerletRbe()) {
    ExecuteVerletRbeOverlap();
    return;
  }

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

void CVFF::ExecuteVerletRbeOverlap() {
  if (!_force_overlap_resources_ready) {
    InitializeForceOverlapResources();
  }

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  // Keep migration/halo/list preparation at the existing completion point.
  // Once Build returns, all streams consume a stable atom layout for the step.
  const auto force_start = DeviceTimingClock::now();
  PrepareLJVerletNeighborList();
  const auto num_atoms = *(_structure_info_data->_num_atoms);

  std::fill_n(_h_overlap_values, 9, rbmd::Real(0));

  CHECK_RUNTIME(EVENT_RECORD(_pair_begin, _pair_stream));
  LaunchLJVerletForce(_pair_stream);
  QueueOverlapPairEnergy();
  CHECK_RUNTIME(EVENT_RECORD(_pair_done, _pair_stream));
  LaunchOverlapBondedForces();

  // CPU sampling/broadcast and stream-local H2D now run while pair and bonded
  // kernels are active. Begin does not resize buffers or synchronize stream 0.
  CHECK_RUNTIME(EVENT_RECORD(_kspace_begin, _kspace_stream));
  _kspace_calculator->BeginRbeForce(_kspace_stream);
  CHECK_RUNTIME(EVENT_RECORD(_rho_ready, _kspace_stream));
  CHECK_RUNTIME(EVENT_SYNC(_rho_ready));
  _kspace_calculator->StartRbeReduction();
  _kspace_calculator->FinishRbeForce(_kspace_stream);
  CHECK_RUNTIME(EVENT_RECORD(_kspace_done, _kspace_stream));

  // One six-value energy reduction replaces up to six blocking collectives.
  // Its host buffers are independent of the far-force output, which is active.
  StartOverlapEnergyReduction();
  CHECK_RUNTIME(STREAM_WAIT_EVENT(_join_stream, _pair_done));
  CHECK_RUNTIME(STREAM_WAIT_EVENT(_join_stream, _kspace_done));
  CHECK_RUNTIME(STREAM_WAIT_EVENT(_join_stream, _bonded_done));
  SumForces(_join_stream);
  CHECK_RUNTIME(EVENT_RECORD(_forces_joined, _join_stream));
  FinishOverlapEnergyReduction();
  CHECK_RUNTIME(EVENT_SYNC(_forces_joined));

  _kspace_calculator->FinalizeRbeForce();
  _kspace_calculator->EvaluatePotentialEnergy();
  RecordOverlapTiming("Pair-OverlapSpan", _pair_begin, _pair_done);
  RecordOverlapTiming("Kspace-OverlapSpan", _kspace_begin, _kspace_done);
  RecordOverlapTiming("Bonded-OverlapSpan", _bonded_begin, _bonded_done);
  const std::chrono::duration<double> force_time =
      DeviceTimingClock::now() - force_start;
  TimingStatistics::Instance().record("Force-Step", force_time.count());

  DebugSyncForceStage("after_short_range_kernel");
  EmitForceStageDebugCsv(
      _device_data, "after_short_range", _device_data->_d_force_ljcoul_x,
      _device_data->_d_force_ljcoul_y, _device_data->_d_force_ljcoul_z,
      num_atoms, _list, linked_cell, force_box);
  DebugSyncForceStage("after_kspace_kernel");
  EmitForceStageDebugCsv(
      _device_data, "after_kspace", _device_data->_d_force_kspace_x,
      _device_data->_d_force_kspace_y, _device_data->_d_force_kspace_z,
      num_atoms);

  DebugSyncForceStage("after_bond_kernel");
  DebugSyncForceStage("after_angle_kernel");
  if (*(_structure_info_data->_num_dihedrals)) {
    DebugSyncForceStage("after_dihedral_kernel");
  }
  if (*(_structure_info_data->_num_impropers)) {
    DebugSyncForceStage("after_improper_kernel");
  }

  DebugSyncForceStage("after_sum_forces_kernel");
  DebugForceStageMarker("before_emit_after_sum_forces_csv");
  EmitForceStageDebugCsv(_device_data, "after_sum_forces",
                         _device_data->_d_fx, _device_data->_d_fy,
                         _device_data->_d_fz, num_atoms);
  DebugForceStageMarker("after_emit_after_sum_forces_csv");

  DebugForceStageMarker("before_evaluate_potential_energy");
  EvaluatePotentialEnergy();
  DebugForceStageMarker("after_evaluate_potential_energy");
}

void CVFF::ExecuteRblRbeOverlap() {
  if (!_force_overlap_resources_ready) {
    InitializeForceOverlapResources();
  }

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  const auto force_start = DeviceTimingClock::now();
  std::fill_n(_h_overlap_values, 9, rbmd::Real(0));
  bool kspace_started = false;
  auto launch_kspace = [&] {
    CHECK_RUNTIME(STREAM_WAIT_EVENT(_kspace_stream, _atoms_ready));
    CHECK_RUNTIME(EVENT_RECORD(_kspace_begin, _kspace_stream));
    _kspace_calculator->BeginRbeForce(_kspace_stream);
    CHECK_RUNTIME(EVENT_RECORD(_rho_ready, _kspace_stream));
    kspace_started = true;
  };
  auto launch_native = [&] {
    // Reuse keeps native positions/charges and allocation stable. Forward
    // exchange writes only validated ghost indices and synchronizes stream 0.
    CHECK_RUNTIME(EVENT_RECORD(_atoms_ready, nullptr));
    launch_kspace();
  };
  auto launch_independent = [&] {
    // Bonded forces can read ghosts, so they start only after halo completion
    // and the ID map update. Re-recording is safe: prior waits captured the
    // earlier native-ready event before coordinate forwarding was issued.
    CHECK_RUNTIME(EVENT_RECORD(_atoms_ready, nullptr));
    CHECK_RUNTIME(STREAM_WAIT_EVENT(_bonded_stream, _atoms_ready));
    LaunchOverlapBondedForces();
    if (!kspace_started) launch_kspace();
    // Start the collective before candidate construction/sampling. Every rank
    // invokes this callback exactly once, before any subsequent builder
    // collective, so the outstanding rho reduction has a consistent order.
    CHECK_RUNTIME(EVENT_SYNC(_rho_ready));
    _kspace_calculator->StartRbeReduction();
  };
  if (RblEnergyEnabled()) {
    // The energy path builds both full and sampled lists. Keep their shared
    // layout preparation together until both builders expose this boundary.
    PrepareRblNeighborLists();
    launch_independent();
  } else {
    const auto neighbor_start = StartDeviceTiming();
    _rbl_list = _rbl_neighbor_list_builder->BuildWithStableAtoms(
        launch_independent, launch_native);
    CHECK_RUNTIME(STREAM_SYNC(nullptr));
    const std::chrono::duration<double> neighbor_time =
        DeviceTimingClock::now() - neighbor_start;
    TimingStatistics::Instance().record("Neighbor-List", neighbor_time.count());
  }
  const auto num_atoms = *(_structure_info_data->_num_atoms);

  CHECK_RUNTIME(EVENT_RECORD(_pair_begin, _pair_stream));
  LaunchRblForce(_pair_stream);
  CHECK_RUNTIME(MEMCPY_ASYNC(
      _h_overlap_values, thrust::raw_pointer_cast(_d_rbl_force_sum.data()),
      3 * sizeof(rbmd::Real), D2H, _pair_stream));
  CHECK_RUNTIME(EVENT_RECORD(_rbl_correction_ready, _pair_stream));
  if (RblEnergyEnabled()) {
    LaunchRblEnergy(_pair_stream);
    QueueOverlapPairEnergy();
  }
  CHECK_RUNTIME(EVENT_RECORD(_pair_done, _pair_stream));

  // Issue the correction early only if its producer is already ready.
  // Waiting for pair here would unnecessarily delay the independent far-force
  // launch. Collective order remains rho -> correction -> energy on every rank.
  const auto correction_status = EVENT_QUERY(_rbl_correction_ready);
  if (correction_status != SUCCESS && correction_status != EVENT_NOT_READY) {
    throw std::runtime_error(GETERRORSTRING(correction_status));
  }
  const bool correction_started = correction_status == SUCCESS;
  if (correction_started) StartOverlapRblCorrection();
  _kspace_calculator->FinishRbeForce(_kspace_stream);
  CHECK_RUNTIME(EVENT_RECORD(_kspace_done, _kspace_stream));
  if (!correction_started) StartOverlapRblCorrection();
  StartOverlapEnergyReduction();
  FinishOverlapRblCorrection();

  CHECK_RUNTIME(STREAM_WAIT_EVENT(_join_stream, _pair_done));
  CHECK_RUNTIME(STREAM_WAIT_EVENT(_join_stream, _kspace_done));
  CHECK_RUNTIME(STREAM_WAIT_EVENT(_join_stream, _bonded_done));
  SumForces(_join_stream);
  CHECK_RUNTIME(EVENT_RECORD(_forces_joined, _join_stream));
  FinishOverlapEnergyReduction();
  CHECK_RUNTIME(EVENT_SYNC(_forces_joined));

  _kspace_calculator->FinalizeRbeForce();
  _kspace_calculator->EvaluatePotentialEnergy();
  RecordOverlapTiming("Pair-OverlapSpan", _pair_begin, _rbl_correction_ready);
  RecordOverlapTiming("Kspace-OverlapSpan", _kspace_begin, _kspace_done);
  RecordOverlapTiming("Bonded-OverlapSpan", _bonded_begin, _bonded_done);
  const std::chrono::duration<double> force_time =
      DeviceTimingClock::now() - force_start;
  TimingStatistics::Instance().record("Force-Step", force_time.count());

  DebugSyncForceStage("after_short_range_kernel");
  EmitForceStageDebugCsv(
      _device_data, "after_short_range", _device_data->_d_force_ljcoul_x,
      _device_data->_d_force_ljcoul_y, _device_data->_d_force_ljcoul_z,
      num_atoms, _rbl_list, linked_cell, force_box);
  DebugSyncForceStage("after_kspace_kernel");
  EmitForceStageDebugCsv(
      _device_data, "after_kspace", _device_data->_d_force_kspace_x,
      _device_data->_d_force_kspace_y, _device_data->_d_force_kspace_z,
      num_atoms);
  DebugSyncForceStage("after_bond_kernel");
  DebugSyncForceStage("after_angle_kernel");
  if (*(_structure_info_data->_num_dihedrals)) {
    DebugSyncForceStage("after_dihedral_kernel");
  }
  if (*(_structure_info_data->_num_impropers)) {
    DebugSyncForceStage("after_improper_kernel");
  }
  DebugSyncForceStage("after_sum_forces_kernel");

  DebugForceStageMarker("before_emit_after_sum_forces_csv");
  EmitForceStageDebugCsv(_device_data, "after_sum_forces",
                         _device_data->_d_fx, _device_data->_d_fy,
                         _device_data->_d_fz, num_atoms);
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
  PrepareRblNeighborLists();
  rbmd::revision::WorkflowTimer revision_pair_wall("Pair-Workflow-Wall");

  const auto start_rbl_force = StartDeviceTiming();
  const auto start_pair_kernel = StartDetailedDeviceTiming();
  LaunchRblForce(nullptr);
  RecordDetailedDeviceTiming("RBL-PairKernel", start_pair_kernel);
  RecordDeviceTiming("Short-Range", start_rbl_force);
  FinalizeRblCorrection();

  if (RblEnergyEnabled()) {
    LaunchRblEnergy(nullptr);
    CHECK_RUNTIME(STREAM_SYNC(nullptr));
    FinalizeRblEnergy();
  }
}

void CVFF::PrepareRblNeighborLists() {
  const auto start = StartDeviceTiming();
  if (RblEnergyEnabled()) {
    _list = _neighbor_list_builder->Build();
    // The full list owns this step's migration/halo preparation. Build the
    // RBL list from the same particle storage, without a second exchange.
    _rbl_list = _rbl_neighbor_list_builder->BuildFromCurrentGhosts();
  } else {
    _rbl_list = _rbl_neighbor_list_builder->Build();
  }
  RecordDeviceTiming("Neighbor-List", start);
}

void CVFF::LaunchRblForce(STREAM_T stream) {
  const auto r_core =
      DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
          "r_core", "hyper_parameters", "neighbor");

  const auto neighbor_sample_num =
      DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
          "neighbor_sample_num", "hyper_parameters", "neighbor");

  const auto num_atoms = *(_structure_info_data->_num_atoms);
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
  const auto rbl_candidate = _rbl_list->_rbl_candidate_list;
  const bool fused_candidate_sampling =
      _rbl_list->_rbl_fused_sampling && static_cast<bool>(rbl_candidate);
  const rbmd::Id* candidate_start = fused_candidate_sampling
      ? thrust::raw_pointer_cast(rbl_candidate->_start_idx.data())
      : nullptr;
  const rbmd::Id* candidate_end = fused_candidate_sampling
      ? thrust::raw_pointer_cast(rbl_candidate->_end_idx.data())
      : nullptr;
  const rbmd::Id* candidate_num = fused_candidate_sampling
      ? thrust::raw_pointer_cast(rbl_candidate->_d_neighbor_num.data())
      : nullptr;
  const rbmd::Id* candidate_neighbors = fused_candidate_sampling
      ? thrust::raw_pointer_cast(rbl_candidate->_d_neighbors.data())
      : nullptr;

  rbmd::revision::GpuTimer revision_pair("Pair-KernelGPU", stream);
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
      thrust::raw_pointer_cast(_rbl_list->_d_random_neighbor_weight.data()),
      fused_candidate_sampling, candidate_start, candidate_end,
      candidate_num, candidate_neighbors,
      fused_candidate_sampling ? rbl_candidate->_max_neighbor_capacity : 0,
      _rbl_list->_rbl_candidate_implicit_offsets,
      _rbl_list->_rbl_sample_epoch,
      thrust::raw_pointer_cast(_d_rbl_coulomb_coeff.data()),
      _rbl_table_r2_min, _rbl_table_scale, _rbl_table_intervals,
      thrust::raw_pointer_cast(_d_rbl_core_taylor.data()),
      d_nspecial, d_special, maxspecial, sbw.w12, sbw.w13, sbw.w14,
      thrust::raw_pointer_cast(_device_data->_d_charge.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()),
      thrust::raw_pointer_cast(_d_rbl_force_sum.data()),
      PairThreadsPerParticle(
          "RBMD_RBL_PAIR_TPP",
          _rbl_list->_max_neighbor_capacity + neighbor_sample_num),
      stream);
  revision_pair.Stop();
}

void CVFF::FinalizeRblCorrection() {
  const auto start_local_reduction = DeviceTimingClock::now();
  CHECK_RUNTIME(MEMCPY(
      _h_rbl_force_sum.data(),
      thrust::raw_pointer_cast(_d_rbl_force_sum.data()),
      3 * sizeof(rbmd::Real), D2H));
  RecordDetailedHostTiming("RBL-LocalReduction", start_local_reduction);

  const auto start_mpi_reduction = DeviceTimingClock::now();
  rbmd::Real global_sum[3] = {_h_rbl_force_sum[0], _h_rbl_force_sum[1],
                              _h_rbl_force_sum[2]};
  AllReduceRealBufferInPlace(global_sum, 3);
  _corr_value_x = global_sum[0] / _global_num_atoms;
  _corr_value_y = global_sum[1] / _global_num_atoms;
  _corr_value_z = global_sum[2] / _global_num_atoms;
  RecordDetailedHostTiming("RBL-MPIAllreduce", start_mpi_reduction);
}

void CVFF::ComputeLJVerlet()
{
  PrepareLJVerletNeighborList();
  rbmd::revision::WorkflowTimer revision_pair_wall("Pair-Workflow-Wall");

  const auto start_verlet_force = StartDeviceTiming();
  const auto start_vl_pair_kernel = StartDetailedDeviceTiming();
  LaunchLJVerletForce(nullptr);
  RecordDetailedDeviceTiming("VL-PairKernel", start_vl_pair_kernel);
  RecordDeviceTiming("Short-Range", start_verlet_force);
  DebugSyncForceStage("verlet_short_range_pre_d2h");
  FinalizeLJVerletForce();
}

void CVFF::PrepareLJVerletNeighborList() {
  const auto start = StartDeviceTiming();
  _list = _neighbor_list_builder->Build();
  RecordDeviceTiming("Neighbor-List", start);
}

void CVFF::LaunchLJVerletForce(STREAM_T stream) {
  CHECK_RUNTIME(MEMSET_ASYNC(
      thrust::raw_pointer_cast(_d_total_evdwl.data()), 0,
      sizeof(rbmd::Real), stream));
  CHECK_RUNTIME(MEMSET_ASYNC(
      thrust::raw_pointer_cast(_d_total_ecoul.data()), 0,
      sizeof(rbmd::Real), stream));

  const auto num_atoms = *(_structure_info_data->_num_atoms);
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
  rbmd::revision::GpuTimer revision_pair("Pair-KernelGPU", stream);
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
                  thrust::raw_pointer_cast(_d_total_ecoul.data()),
                  PairThreadsPerParticle("RBMD_VL_PAIR_TPP",
                                         _list->_max_neighbor_capacity),
                  stream);
  revision_pair.Stop();

  op::ReduceVirialOp<device::DEVICE_GPU>()(
      num_atoms, num_atoms,
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
      thrust::raw_pointer_cast(_device_data->_d_virial_lj.data()), stream);
}

void CVFF::FinalizeLJVerletForce() {
  rbmd::Real local_vdwl = 0;
  rbmd::Real local_coul = 0;
  CHECK_RUNTIME(MEMCPY(
      &local_vdwl, thrust::raw_pointer_cast(_d_total_evdwl.data()),
      sizeof(rbmd::Real), D2H));
  CHECK_RUNTIME(MEMCPY(
      &local_coul, thrust::raw_pointer_cast(_d_total_ecoul.data()),
      sizeof(rbmd::Real), D2H));
  _e_vdwl = GetGlobalRealSum(local_vdwl);
  _e_coul = GetGlobalRealSum(local_coul);
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
                  _geometric_lj_mixing,
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
  //   LinkedCellLocator::GetInstance().GetLinkedCell()->AtomIndexMap();
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
  _list = _neighbor_list_builder->Build();
  LaunchRblEnergy(nullptr);
  CHECK_RUNTIME(STREAM_SYNC(nullptr));
  FinalizeRblEnergy();
}

void CVFF::LaunchRblEnergy(STREAM_T stream) {
  ClearDeviceVectorAsync(_d_total_evdwl, stream);
  ClearDeviceVectorAsync(_d_total_ecoul, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_lj, stream);

  const auto num_atoms = *(_structure_info_data->_num_atoms);
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
                _geometric_lj_mixing,
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
                thrust::raw_pointer_cast(_d_total_ecoul.data()), stream);

  op::ReduceVirialOp<device::DEVICE_GPU>()(
      num_atoms, num_atoms,
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
      thrust::raw_pointer_cast(_device_data->_d_virial_lj.data()), stream);
}

void CVFF::FinalizeRblEnergy() {
  rbmd::Real local_vdwl = 0;
  rbmd::Real local_coul = 0;
  CHECK_RUNTIME(MEMCPY(
      &local_vdwl, thrust::raw_pointer_cast(_d_total_evdwl.data()),
      sizeof(rbmd::Real), D2H));
  CHECK_RUNTIME(MEMCPY(
      &local_coul, thrust::raw_pointer_cast(_d_total_ecoul.data()),
      sizeof(rbmd::Real), D2H));
  _e_vdwl = GetGlobalRealSum(local_vdwl);
  _e_coul = GetGlobalRealSum(local_coul);
}

void CVFF::SumForces(STREAM_T stream)
{
  const bool apply_rbl_correction = "RBL" == _neighbor_type;
  const rbmd::Id num_atoms = *(_structure_info_data->_num_atoms);
  rbmd::revision::GpuTimer revision_sum("Force-AssembleGPU", stream);
  op::SumRblForcesOp<device::DEVICE_GPU>()(
      num_atoms,
      apply_rbl_correction ? _corr_value_x : rbmd::Real(0),
      apply_rbl_correction ? _corr_value_y : rbmd::Real(0),
      apply_rbl_correction ? _corr_value_z : rbmd::Real(0),
      thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_kspace_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_kspace_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_kspace_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_bond_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_angle_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_dihedral_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_x.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_y.data()),
      thrust::raw_pointer_cast(_device_data->_d_force_improper_z.data()),
      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
      thrust::raw_pointer_cast(_device_data->_d_fz.data()), stream);
  revision_sum.Stop();
}

void CVFF::ComputeBondForce(STREAM_T stream, bool defer_finalize)
{
  const auto start = defer_finalize ? DeviceTimingClock::now()
                                    : StartDeviceTiming();

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto& atom_id_to_idx = linked_cell->AtomIndexMap();

  ClearDeviceVectorAsync(_device_data->_d_force_bond_x, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_bond_y, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_bond_z, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_bond_atom, stream);
  ClearDeviceVectorAsync(_d_total_ebond, stream);
  ClearDeviceVectorAsync(_d_total_cbond, stream);

  auto num_bonds = *(_structure_info_data->_num_bonds);
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const rbmd::Id atom_id_to_idx_size =
      static_cast<rbmd::Id>(atom_id_to_idx.count);
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomBondPath()) {
    op::ComputeBondForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_bond_atom.data()),
      thrust::raw_pointer_cast(_d_total_ebond.data()),
      thrust::raw_pointer_cast(_d_total_cbond.data()), stream);
  } else {
    op::ComputeBondForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_bonds,atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_ebond.data()), stream);
  }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
  thrust::raw_pointer_cast(_device_data->_d_flat_virial_bond_atom.data()),
  thrust::raw_pointer_cast(_device_data->_d_virial_bond.data()), stream);

  if (!defer_finalize) {
    RecordDeviceTiming("Bond", start);
    DebugSyncForceStage("bond_pre_d2h");
    thrust::host_vector<rbmd::Real> h_total_ebond(_d_total_ebond);
    _e_bond = GetGlobalRealSum(h_total_ebond[0]);
    ThermoStats::Instance().AddThermoData("bond", _e_bond);
  }
}

void CVFF::ComputeAngleForce(STREAM_T stream, bool defer_finalize)
{
  const auto start = defer_finalize ? DeviceTimingClock::now()
                                    : StartDeviceTiming();
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto& atom_id_to_idx = linked_cell->AtomIndexMap();

  ClearDeviceVectorAsync(_device_data->_d_force_angle_x, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_angle_y, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_angle_z, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_angle_atom, stream);
  ClearDeviceVectorAsync(_d_total_eangle, stream);
  ClearDeviceVectorAsync(_d_total_cangle, stream);

  auto num_angles = *(_structure_info_data->_num_angles);
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomAnglePath()) {
    op::ComputeAngleForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_device_data->_d_flat_virial_angle_atom.data()),
      thrust::raw_pointer_cast(_d_total_eangle.data()),
      thrust::raw_pointer_cast(_d_total_cangle.data()), stream);
  } else {
    op::ComputeAngleForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_angles,
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_eangle.data()), stream);
  }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_angle_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_angle.data()), stream);

  if (!defer_finalize) {
    RecordDeviceTiming("Angle", start);
    DebugSyncForceStage("angle_pre_d2h");
    thrust::host_vector<rbmd::Real> h_total_eangle(_d_total_eangle);
    _e_angle = GetGlobalRealSum(h_total_eangle[0]);
    ThermoStats::Instance().AddThermoData("angle", _e_angle);
  }
}

void CVFF::ComputeDihedralForce(STREAM_T stream, bool defer_finalize)
{
  auto dihedral_type = DataManager::getInstance().getConfigData()->Get
  <std::string>("dihedral_type", "hyper_parameters", "force_field");

  if (dihedral_type == "harmonic") {
    DihedralHarmonic(stream, defer_finalize);
  }
  else if (dihedral_type == "opls") {
    DihedralOPLS(stream, defer_finalize);
  }
  else if (dihedral_type == "fourier") {
    DihedralFourier(stream, defer_finalize);
  }
  if (!defer_finalize) {
    ThermoStats::Instance().AddThermoData("dihedral", _e_dihedral);
  }
}

void CVFF::DihedralHarmonic(STREAM_T stream, bool defer_finalize) {
  const auto start = defer_finalize ? DeviceTimingClock::now()
                                    : StartDeviceTiming();
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_x, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_y, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_z, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_dihedral_atom, stream);
  ClearDeviceVectorAsync(_d_total_edihedral, stream);
  ClearDeviceVectorAsync(_d_total_cdihedral, stream);

  //thrust::device_vector<int4> dihedral_list;
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto& atom_id_to_idx = linked_cell->AtomIndexMap();

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_dihedrals = *(_structure_info_data->_num_dihedrals);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomDihedralPath()) {
    op::ComputeDihedralForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_edihedral.data()),
      thrust::raw_pointer_cast(_d_total_cdihedral.data()), stream);
  } else {
    op::ComputeDihedralForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_dihedrals,
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_edihedral.data()), stream);
  }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_dihedral.data()), stream);

  if (!defer_finalize) {
    RecordDeviceTiming("Dihedral", start);
    thrust::host_vector<rbmd::Real> h_total_edihedral(_d_total_edihedral);
    _e_dihedral = GetGlobalRealSum(h_total_edihedral[0]);
  }
}

void CVFF::DihedralOPLS(STREAM_T stream, bool defer_finalize) {
  const auto start = defer_finalize ? DeviceTimingClock::now()
                                    : StartDeviceTiming();
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_x, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_y, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_z, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_dihedral_atom, stream);
  ClearDeviceVectorAsync(_d_total_edihedral, stream);
  ClearDeviceVectorAsync(_d_total_cdihedral, stream);

  //thrust::device_vector<int4> dihedral_list;
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto& atom_id_to_idx = linked_cell->AtomIndexMap();

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_dihedrals = *(_structure_info_data->_num_dihedrals);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomDihedralPath()) {
    op::ComputeDihedralOPLSForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_edihedral.data()),
      thrust::raw_pointer_cast(_d_total_cdihedral.data()), stream);
  } else {
    op::ComputeDihedralOPLSForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_dihedrals,
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_edihedral.data()), stream);
  }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_dihedral.data()), stream);

  if (!defer_finalize) {
    RecordDeviceTiming("Dihedral", start);
    thrust::host_vector<rbmd::Real> h_total_edihedral(_d_total_edihedral);
    _e_dihedral = GetGlobalRealSum(h_total_edihedral[0]);
  }
}

void CVFF::DihedralFourier(STREAM_T stream, bool defer_finalize) {
  const auto start = defer_finalize ? DeviceTimingClock::now()
                                    : StartDeviceTiming();
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_x, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_y, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_dihedral_z, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_dihedral_atom, stream);
  ClearDeviceVectorAsync(_d_total_edihedral, stream);
  ClearDeviceVectorAsync(_d_total_cdihedral, stream);

  //thrust::device_vector<int4> dihedral_list;
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto& atom_id_to_idx = linked_cell->AtomIndexMap();

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_dihedrals = *(_structure_info_data->_num_dihedrals);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomDihedralPath()) {
    op::ComputeDihedralFourierForcePerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_edihedral.data()),
      thrust::raw_pointer_cast(_d_total_cdihedral.data()), stream);
  } else {
    op::ComputeDihedralFourierForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_dihedrals,
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_edihedral.data()), stream);
  }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_dihedral_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_dihedral.data()), stream);

  if (!defer_finalize) {
    RecordDeviceTiming("Dihedral", start);
    thrust::host_vector<rbmd::Real> h_total_edihedral(_d_total_edihedral);
    _e_dihedral = GetGlobalRealSum(h_total_edihedral[0]);
  }
}

void CVFF::ComputeImproperForce(STREAM_T stream, bool defer_finalize)
{
  auto improper_type = DataManager::getInstance().getConfigData()->Get
<std::string>("improper_type", "hyper_parameters", "force_field");

  if (improper_type == "harmonic") {
    ImproperHarmonic(stream, defer_finalize);
  }
  else if (improper_type == "cvff") {
    ImproperCVFF(stream, defer_finalize);
  }

  if (!defer_finalize) {
    ThermoStats::Instance().AddThermoData("improper", _e_improper);
  }
}

void CVFF::ImproperHarmonic(STREAM_T stream, bool defer_finalize) {
  const auto start = defer_finalize ? DeviceTimingClock::now()
                                    : StartDeviceTiming();
  ClearDeviceVectorAsync(_device_data->_d_force_improper_x, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_improper_y, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_improper_z, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_improper_atom, stream);
  ClearDeviceVectorAsync(_d_total_eimproper, stream);
  ClearDeviceVectorAsync(_d_total_cimproper, stream);

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto& atom_id_to_idx = linked_cell->AtomIndexMap();

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_impropers = *(_structure_info_data->_num_impropers);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomImproperPath()) {
    op::ComputeImproperHarmonicPerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_eimproper.data()),
      thrust::raw_pointer_cast(_d_total_cimproper.data()), stream);
  } else {
    op::ComputeImproperHarmonicForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_impropers,
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_eimproper.data()), stream);
  }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_improper.data()), stream);

  if (!defer_finalize) {
    RecordDeviceTiming("Improper", start);
    thrust::host_vector<rbmd::Real> h_total_eimproper(_d_total_eimproper);
    _e_improper = GetGlobalRealSum(h_total_eimproper[0]);
  }
}

void CVFF::ImproperCVFF(STREAM_T stream, bool defer_finalize)
{
  const auto start = defer_finalize ? DeviceTimingClock::now()
                                    : StartDeviceTiming();
  ClearDeviceVectorAsync(_device_data->_d_force_improper_x, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_improper_y, stream);
  ClearDeviceVectorAsync(_device_data->_d_force_improper_z, stream);
  ClearDeviceVectorAsync(_device_data->_d_flat_virial_improper_atom, stream);
  ClearDeviceVectorAsync(_d_total_eimproper, stream);
  ClearDeviceVectorAsync(_d_total_cimproper, stream);

  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto& atom_id_to_idx = linked_cell->AtomIndexMap();

  auto num_atoms = *(_structure_info_data->_num_atoms);
  auto num_impropers = *(_structure_info_data->_num_impropers);
  const rbmd::Id num_total_atoms =
      linked_cell ? linked_cell->_total_atoms_num : num_atoms;
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (UsePerAtomImproperPath()) {
    op::ComputeImproperCVFFPerAtomOp<device::DEVICE_GPU>()(
      force_box, num_atoms, num_total_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_eimproper.data()),
      thrust::raw_pointer_cast(_d_total_cimproper.data()), stream);
  } else {
    op::ComputeImproperCVFFForceOp<device::DEVICE_GPU>()(
      force_box,num_atoms,num_impropers,
      atom_id_to_idx,
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
      thrust::raw_pointer_cast(_d_total_eimproper.data()), stream);
  }

  op::ReduceVirialOp<device::DEVICE_GPU>()(num_atoms,num_atoms,
thrust::raw_pointer_cast(_device_data->_d_flat_virial_improper_atom.data()),
thrust::raw_pointer_cast(_device_data->_d_virial_improper.data()), stream);

  if (!defer_finalize) {
    RecordDeviceTiming("Improper", start);
    thrust::host_vector<rbmd::Real> h_total_eimproper(_d_total_eimproper);
    _e_improper = GetGlobalRealSum(h_total_eimproper[0]);
  }
}

void CVFF::FinalizeBondedForces() {
  thrust::host_vector<rbmd::Real> h_bond(_d_total_ebond);
  thrust::host_vector<rbmd::Real> h_angle(_d_total_eangle);
  _e_bond = GetGlobalRealSum(h_bond[0]);
  _e_angle = GetGlobalRealSum(h_angle[0]);

  if (*(_structure_info_data->_num_dihedrals)) {
    thrust::host_vector<rbmd::Real> h_dihedral(_d_total_edihedral);
    _e_dihedral = GetGlobalRealSum(h_dihedral[0]);
  } else {
    _e_dihedral = 0;
  }
  if (*(_structure_info_data->_num_impropers)) {
    thrust::host_vector<rbmd::Real> h_improper(_d_total_eimproper);
    _e_improper = GetGlobalRealSum(h_improper[0]);
  } else {
    _e_improper = 0;
  }

  ThermoStats::Instance().AddThermoData("bond", _e_bond);
  ThermoStats::Instance().AddThermoData("angle", _e_angle);
  if (*(_structure_info_data->_num_dihedrals)) {
    ThermoStats::Instance().AddThermoData("dihedral", _e_dihedral);
  }
  if (*(_structure_info_data->_num_impropers)) {
    ThermoStats::Instance().AddThermoData("improper", _e_improper);
  }
}

void CVFF::WriteForceSnapshot() const {
  static const long long requested_step = [] {
    const char* value = std::getenv("RBMD_FORCE_SNAPSHOT_STEP");
    if (!value || !*value) return -1LL;
    char* end = nullptr;
    const long long step = std::strtoll(value, &end, 10);
    return end && *end == '\0' && step >= 0 ? step : -1LL;
  }();
  if (requested_step < 0 || requested_step != test_current_step) return;
  // Called only after the force join and all energy requests have completed.
  // The default-off snapshot makes serial/concurrent GPU results comparable
  // by global ID without changing kernel launch ordering during computation.
  EnsureDebugLogDirectory();
  const auto n = static_cast<std::size_t>(*(_structure_info_data->_num_atoms));
  const std::array<const thrust::device_vector<rbmd::Real>*, 24> arrays{{
      &_device_data->_d_px, &_device_data->_d_py, &_device_data->_d_pz,
      &_device_data->_d_fx, &_device_data->_d_fy, &_device_data->_d_fz,
      &_device_data->_d_force_ljcoul_x, &_device_data->_d_force_ljcoul_y,
      &_device_data->_d_force_ljcoul_z,
      &_device_data->_d_force_kspace_x, &_device_data->_d_force_kspace_y,
      &_device_data->_d_force_kspace_z,
      &_device_data->_d_force_bond_x, &_device_data->_d_force_bond_y,
      &_device_data->_d_force_bond_z,
      &_device_data->_d_force_angle_x, &_device_data->_d_force_angle_y,
      &_device_data->_d_force_angle_z,
      &_device_data->_d_force_dihedral_x, &_device_data->_d_force_dihedral_y,
      &_device_data->_d_force_dihedral_z,
      &_device_data->_d_force_improper_x, &_device_data->_d_force_improper_y,
      &_device_data->_d_force_improper_z}};
  std::vector<thrust::host_vector<rbmd::Real>> values;
  for (const auto* array : arrays) {
    if (array->size() < n) throw std::runtime_error("Force snapshot size mismatch");
    values.emplace_back(array->begin(), array->begin() + n);
  }
  thrust::host_vector<rbmd::Id> ids(_device_data->_d_atoms_id.begin(),
                                   _device_data->_d_atoms_id.begin() + n);
  const std::string prefix = "logs/debug/force_snapshot_step" +
      std::to_string(test_current_step) + "_rank" + std::to_string(CurrentRank());
  std::ofstream forces(prefix + ".csv");
  forces << std::scientific << std::setprecision(std::numeric_limits<rbmd::Real>::max_digits10);
  forces << "gid,x,y,z,total_x,total_y,total_z,pair_x,pair_y,pair_z,"
            "kspace_x,kspace_y,kspace_z,bond_x,bond_y,bond_z,angle_x,angle_y,angle_z,"
            "dihedral_x,dihedral_y,dihedral_z,improper_x,improper_y,improper_z\n";
  for (std::size_t i = 0; i < n; ++i) {
    forces << ids[i];
    for (const auto& component : values) forces << ',' << component[i];
    forces << '\n';
  }
  std::ofstream totals(prefix + ".totals.csv");
  totals << std::scientific << std::setprecision(std::numeric_limits<rbmd::Real>::max_digits10);
  totals << "kind,name,values\n";
  auto virial = [&](const char* name, const thrust::device_vector<rbmd::Real>& input) {
    thrust::host_vector<rbmd::Real> host(input);
    totals << "virial," << name;
    for (auto v : host) totals << ',' << v;
    totals << '\n';
  };
  virial("pair", _device_data->_d_virial_lj);
  virial("kspace", _device_data->_d_virial_kspace);
  virial("bond", _device_data->_d_virial_bond);
  virial("angle", _device_data->_d_virial_angle);
  virial("dihedral", _device_data->_d_virial_dihedral);
  virial("improper", _device_data->_d_virial_improper);
  totals << "energy,global," << _e_vdwl << ',' << _e_coul << ',' << _e_kspace
         << ',' << _e_bond << ',' << _e_angle << ',' << _e_dihedral
         << ',' << _e_improper << '\n';
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
  WriteForceSnapshot();

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
