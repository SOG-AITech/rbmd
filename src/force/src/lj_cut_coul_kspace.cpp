#include "common/revision_timing.h"
#include "lj_cut_coul_kspace.h"
#include "force_box_selector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <thrust/find.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "../../common/device_types.h"
#include "../../common/rbmd_define.h"
#include "../../common/types.h"
#include "../../common/unit_factor.h"
#include "force_op/force_op.h"
#include "lj_op/lj_op.h"
#include "lj_cut_coul_kspace_op/lj_cut_coul_kspace_op.h"
#include "../common/RBEPSample.h"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"
#include "neighbor_list/include/neighbor_list_builder/full_neighbor_list_builder.h"
#include "neighbor_list/include/neighbor_list_builder/rbl_full_neighbor_list_builder.h"
// #include <hipcub/hipcub.hpp>
// #include <hipcub/backend/rocprim/block/block_reduce.hpp>
#include "common/math_utils.h"
#include "common/mpi_reduce_helper.hpp"
#include "common/mpi_root_guard.hpp"
#include "common/thermo_stats.hpp"
#include "common/timing_statistics.hpp"
#include "cvff_op.h"
#include "rbsog_level_preset.h"

extern rbmd::Id test_current_step;
extern std::map<std::string, UNIT> unit_factor_map;

namespace {

using RBLForceSum =
    thrust::tuple<rbmd::Real, rbmd::Real, rbmd::Real>;

struct AddRBLForceSum {
  __host__ __device__ RBLForceSum operator()(const RBLForceSum& lhs,
                                             const RBLForceSum& rhs) const {
    return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
                              thrust::get<1>(lhs) + thrust::get<1>(rhs),
                              thrust::get<2>(lhs) + thrust::get<2>(rhs));
  }
};

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void LogLJHipMemory(const char* stage) {
  if (!EnvFlagEnabled("RBMD_HIP_MEMORY_DIAG")) {
    return;
  }
  (void)DEVICESYNC();
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const ERROR_T status = MEMGETINFO(&free_bytes, &total_bytes);
  if (status != SUCCESS) {
    Logger::Instance().info("RBMD-HIP-MEM component=lj stage={} status={}",
                            stage, GETERRORSTRING(status));
    return;
  }
  Logger::Instance().info(
      "RBMD-HIP-MEM component=lj stage={} free_bytes={} used_bytes={} total_bytes={}",
      stage, free_bytes, total_bytes, total_bytes - free_bytes);
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

void EnsurePairSpikeDebugLogDirectory() {
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

struct PairSpikeDebugConfig {
  bool enabled{false};
  int every{1};
};

PairSpikeDebugConfig ParsePairSpikeDebugConfig() {
  PairSpikeDebugConfig cfg;
  cfg.enabled = EnvFlagEnabled("RBMD_DEBUG_PAIR_SPIKE_STATS");
  cfg.every = EnvIntOrDefault("RBMD_DEBUG_PAIR_SPIKE_STATS_EVERY", 1);
  return cfg;
}

const PairSpikeDebugConfig& GetPairSpikeDebugConfig() {
  static const PairSpikeDebugConfig cfg = ParsePairSpikeDebugConfig();
  return cfg;
}

bool ShouldWritePairSpikeDebug(rbmd::Id step) {
  const auto& cfg = GetPairSpikeDebugConfig();
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

struct PairSpikeRow {
  bool valid{false};
  const char* kind{""};
  const char* stage{""};
  rbmd::Id atom_idx{-1};
  rbmd::Id atom_gid{-1};
  rbmd::Id atom_type{-1};
  rbmd::Id atom_cell_idx{-1};
  int atom_cell_is_halo{-1};
  rbmd::Id neighbor_idx{-1};
  rbmd::Id neighbor_gid{-1};
  rbmd::Id neighbor_type{-1};
  const char* neighbor_region{"missing"};
  rbmd::Id neighbor_cell_idx{-1};
  int neighbor_cell_is_halo{-1};
  rbmd::Real distance_sq{0};
  rbmd::Real distance{0};
  rbmd::Real dx{0};
  rbmd::Real dy{0};
  rbmd::Real dz{0};
  rbmd::Real lj_force_scalar{0};
  rbmd::Real coul_force_scalar{0};
  rbmd::Real pair_force_abs{0};
};

void EmitPairSpikeDebugCsv(const std::shared_ptr<DeviceData>& device_data,
                           const std::shared_ptr<NeighborList>& neighbor_list,
                           const std::shared_ptr<LinkedCell>& linked_cell,
                           const Box& force_box, const char* stage,
                           rbmd::Real cut_off, rbmd::Real alpha,
                           rbmd::Real qqr2e, rbmd::Id nlocal) {
  if (!device_data || !neighbor_list || !linked_cell || !stage ||
      stage[0] == '\0' || !ShouldWritePairSpikeDebug(test_current_step)) {
    return;
  }

  const rbmd::Id safe_nlocal = std::max<rbmd::Id>(nlocal, 0);
  const rbmd::Id safe_ntotal = std::max<rbmd::Id>(linked_cell->_total_atoms_num, 0);
  const std::size_t native_size = std::min<std::size_t>(
      static_cast<std::size_t>(safe_nlocal),
      std::min(neighbor_list->_start_idx.size(), neighbor_list->_end_idx.size()));
  const std::size_t total_size = std::min<std::size_t>(
      static_cast<std::size_t>(safe_ntotal),
      std::min(device_data->_d_atoms_id.size(),
               std::min(device_data->_d_atoms_type.size(),
                        std::min(device_data->_d_charge.size(),
                                 std::min(device_data->_d_px.size(),
                                          std::min(device_data->_d_py.size(),
                                                   std::min(device_data->_d_pz.size(),
                                                            linked_cell->_per_atom_cell_id.size())))))));

  if (native_size == 0 || total_size == 0) {
    return;
  }

  thrust::host_vector<rbmd::Id> h_ids(device_data->_d_atoms_id.begin(),
                                      device_data->_d_atoms_id.begin() + total_size);
  thrust::host_vector<rbmd::Id> h_types(device_data->_d_atoms_type.begin(),
                                        device_data->_d_atoms_type.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_charge(device_data->_d_charge.begin(),
                                           device_data->_d_charge.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_px(device_data->_d_px.begin(),
                                       device_data->_d_px.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_py(device_data->_d_py.begin(),
                                       device_data->_d_py.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_pz(device_data->_d_pz.begin(),
                                       device_data->_d_pz.begin() + total_size);
  thrust::host_vector<rbmd::Real> h_sigma = device_data->_d_sigma;
  thrust::host_vector<rbmd::Real> h_eps = device_data->_d_eps;
  thrust::host_vector<rbmd::Id> h_cell_id(linked_cell->_per_atom_cell_id.begin(),
                                          linked_cell->_per_atom_cell_id.begin() + total_size);
  thrust::host_vector<rbmd::Id> h_start(neighbor_list->_start_idx.begin(),
                                        neighbor_list->_start_idx.begin() + native_size);
  thrust::host_vector<rbmd::Id> h_end(neighbor_list->_end_idx.begin(),
                                      neighbor_list->_end_idx.begin() + native_size);
  thrust::host_vector<rbmd::Id> h_neighbors = neighbor_list->_d_neighbors;
  thrust::host_vector<Cell> h_cells = linked_cell->_cells;
  const std::size_t sigma_size = h_sigma.size();
  const std::size_t eps_size = h_eps.size();

  PairSpikeRow min_distance_pair{};
  min_distance_pair.kind = "min_distance_pair";
  min_distance_pair.stage = stage;
  PairSpikeRow max_pair_force{};
  max_pair_force.kind = "max_pair_force";
  max_pair_force.stage = stage;

  const rbmd::Real cut_off_sq = cut_off * cut_off;
  const rbmd::Real sqrt_alpha = std::sqrt(alpha);
  constexpr rbmd::Real kPi = rbmd::Real(3.14159265358979323846);

  auto fill_row = [&](PairSpikeRow& row, rbmd::Id atom_idx, rbmd::Id neighbor_idx,
                      rbmd::Real dx, rbmd::Real dy, rbmd::Real dz,
                      rbmd::Real distance_sq, rbmd::Real distance,
                      rbmd::Real lj_force_scalar, rbmd::Real coul_force_scalar,
                      rbmd::Real pair_force_abs) {
    row.valid = true;
    row.atom_idx = atom_idx;
    row.atom_gid = h_ids[static_cast<std::size_t>(atom_idx)];
    row.atom_type = h_types[static_cast<std::size_t>(atom_idx)];
    row.atom_cell_idx = h_cell_id[static_cast<std::size_t>(atom_idx)];
    if (row.atom_cell_idx >= 0 &&
        static_cast<std::size_t>(row.atom_cell_idx) < h_cells.size()) {
      row.atom_cell_is_halo =
          h_cells[static_cast<std::size_t>(row.atom_cell_idx)]._is_halo ? 1 : 0;
    }
    row.neighbor_idx = neighbor_idx;
    row.neighbor_gid = h_ids[static_cast<std::size_t>(neighbor_idx)];
    row.neighbor_type = h_types[static_cast<std::size_t>(neighbor_idx)];
    row.neighbor_region = AtomRegionName(neighbor_idx, safe_nlocal, safe_ntotal);
    row.neighbor_cell_idx = h_cell_id[static_cast<std::size_t>(neighbor_idx)];
    if (row.neighbor_cell_idx >= 0 &&
        static_cast<std::size_t>(row.neighbor_cell_idx) < h_cells.size()) {
      row.neighbor_cell_is_halo =
          h_cells[static_cast<std::size_t>(row.neighbor_cell_idx)]._is_halo ? 1 : 0;
    }
    row.distance_sq = distance_sq;
    row.distance = distance;
    row.dx = dx;
    row.dy = dy;
    row.dz = dz;
    row.lj_force_scalar = lj_force_scalar;
    row.coul_force_scalar = coul_force_scalar;
    row.pair_force_abs = pair_force_abs;
  };

  for (std::size_t atom_offset = 0; atom_offset < native_size; ++atom_offset) {
    const rbmd::Id atom_idx = static_cast<rbmd::Id>(atom_offset);
    const rbmd::Id begin = ClampNeighborOffset(h_start[atom_offset], h_neighbors.size());
    const rbmd::Id end = ClampNeighborOffset(h_end[atom_offset], h_neighbors.size());
    const rbmd::Id self_type = h_types[atom_offset];
    if (self_type < 0 || static_cast<std::size_t>(self_type) >= sigma_size ||
        static_cast<std::size_t>(self_type) >= eps_size) {
      continue;
    }
    const rbmd::Real sigma_i = h_sigma[static_cast<std::size_t>(self_type)];
    const rbmd::Real eps_i = h_eps[static_cast<std::size_t>(self_type)];
    const rbmd::Real charge_i = h_charge[atom_offset];
    const rbmd::Real x1 = h_px[atom_offset];
    const rbmd::Real y1 = h_py[atom_offset];
    const rbmd::Real z1 = h_pz[atom_offset];

    for (rbmd::Id slot = begin; slot < end; ++slot) {
      const std::size_t neighbor_slot = static_cast<std::size_t>(slot);
      if (neighbor_slot >= h_neighbors.size()) {
        continue;
      }

      const rbmd::Id neighbor_idx = h_neighbors[neighbor_slot];
      if (neighbor_idx < 0 ||
          static_cast<std::size_t>(neighbor_idx) >= total_size ||
          neighbor_idx == atom_idx) {
        continue;
      }

      rbmd::Real dx = h_px[static_cast<std::size_t>(neighbor_idx)] - x1;
      rbmd::Real dy = h_py[static_cast<std::size_t>(neighbor_idx)] - y1;
      rbmd::Real dz = h_pz[static_cast<std::size_t>(neighbor_idx)] - z1;
      MinImageDistance_while(force_box, dx, dy, dz);
      const rbmd::Real distance_sq = dx * dx + dy * dy + dz * dz;
      if (!(distance_sq > EPSILON) || distance_sq >= cut_off_sq) {
        continue;
      }

      const rbmd::Real distance = std::sqrt(distance_sq);
      const rbmd::Id neighbor_type = h_types[static_cast<std::size_t>(neighbor_idx)];
      if (neighbor_type < 0 ||
          static_cast<std::size_t>(neighbor_type) >= sigma_size ||
          static_cast<std::size_t>(neighbor_type) >= eps_size) {
        continue;
      }
      const rbmd::Real sigma_j = h_sigma[static_cast<std::size_t>(neighbor_type)];
      const rbmd::Real eps_j = h_eps[static_cast<std::size_t>(neighbor_type)];
      const rbmd::Real charge_j = h_charge[static_cast<std::size_t>(neighbor_idx)];

      const rbmd::Real eps_ij = std::sqrt(eps_i * eps_j);
      const rbmd::Real sigma_ij = (sigma_i + sigma_j) / rbmd::Real(2);
      const rbmd::Real sigmaij_6 = std::pow(sigma_ij, rbmd::Real(6));
      const rbmd::Real dis_6 = std::pow(distance_sq, rbmd::Real(3));
      const rbmd::Real sigmaij_dis_6 = sigmaij_6 / dis_6;
      const rbmd::Real lj_force_scalar =
          -rbmd::Real(24) * eps_ij *
          ((rbmd::Real(2) * sigmaij_dis_6 - rbmd::Real(1)) * sigmaij_dis_6) /
          distance_sq;

      const rbmd::Real erfc_value = std::erfc(sqrt_alpha * distance);
      const rbmd::Real exp_value = std::exp(-alpha * distance_sq);
      const rbmd::Real coul_near =
          erfc_value / distance_sq +
          rbmd::Real(2) * sqrt_alpha * exp_value /
              (std::sqrt(kPi) * distance);
      const rbmd::Real coul_force_scalar =
          -qqr2e * charge_i * charge_j * coul_near / distance;
      const rbmd::Real pair_force_abs =
          std::abs(lj_force_scalar + coul_force_scalar) * distance;

      if (!min_distance_pair.valid ||
          distance_sq < min_distance_pair.distance_sq) {
        fill_row(min_distance_pair, atom_idx, neighbor_idx, dx, dy, dz,
                 distance_sq, distance, lj_force_scalar, coul_force_scalar,
                 pair_force_abs);
      }
      if (!max_pair_force.valid ||
          pair_force_abs > max_pair_force.pair_force_abs) {
        fill_row(max_pair_force, atom_idx, neighbor_idx, dx, dy, dz,
                 distance_sq, distance, lj_force_scalar, coul_force_scalar,
                 pair_force_abs);
      }
    }
  }

  if (!min_distance_pair.valid && !max_pair_force.valid) {
    return;
  }

  EnsurePairSpikeDebugLogDirectory();
  const int rank = CurrentRank();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/lj_pair_spike_stats_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,stage,kind,nlocal,ntotal,atom_idx,atom_gid,atom_type,"
           "atom_cell_idx,atom_cell_is_halo,neighbor_idx,neighbor_gid,"
           "neighbor_type,neighbor_region,neighbor_cell_idx,"
           "neighbor_cell_is_halo,distance_sq,distance,dx,dy,dz,"
           "lj_force_scalar,coul_force_scalar,pair_force_abs\n";
  }

  auto append_row = [&](const PairSpikeRow& row) {
    if (!row.valid) {
      return;
    }
    csv << test_current_step << "," << rank << "," << row.stage << ","
        << row.kind << "," << safe_nlocal << "," << safe_ntotal << ","
        << row.atom_idx << "," << row.atom_gid << "," << row.atom_type << ","
        << row.atom_cell_idx << "," << row.atom_cell_is_halo << ","
        << row.neighbor_idx << "," << row.neighbor_gid << ","
        << row.neighbor_type << "," << row.neighbor_region << ","
        << row.neighbor_cell_idx << "," << row.neighbor_cell_is_halo << ","
        << row.distance_sq << "," << row.distance << "," << row.dx << ","
        << row.dy << "," << row.dz << "," << row.lj_force_scalar << ","
        << row.coul_force_scalar << "," << row.pair_force_abs << "\n";
  };

  append_row(min_distance_pair);
  append_row(max_pair_force);
}

struct NeighborRowAudit {
  bool valid{false};
  rbmd::Id count{-1};
  rbmd::Id written{-1};
  bool contains{false};
};

bool ShouldWritePairMembershipDebug(rbmd::Id step) {
  static const bool enabled =
      EnvFlagEnabled("RBMD_DEBUG_PAIR_MEMBERSHIP");
  static const int first_step =
      EnvIntOrDefault("RBMD_DEBUG_PAIR_MEMBERSHIP_START", 190);
  static const int last_step =
      EnvIntOrDefault("RBMD_DEBUG_PAIR_MEMBERSHIP_END", 207);
  return enabled && step >= first_step && step <= last_step;
}

template <typename DeviceVector>
rbmd::Id FindGidIndex(const DeviceVector& ids, rbmd::Id atom_count,
                      rbmd::Id target_gid) {
  const std::size_t limit = std::min<std::size_t>(
      static_cast<std::size_t>(std::max<rbmd::Id>(atom_count, 0)),
      ids.size());
  const auto end = ids.begin() + limit;
  const auto found = thrust::find(ids.begin(), end, target_gid);
  return found == end ? rbmd::Id(-1)
                      : static_cast<rbmd::Id>(found - ids.begin());
}

NeighborRowAudit InspectNeighborRow(
    const std::shared_ptr<NeighborList>& list, rbmd::Id center_idx,
    rbmd::Id partner_idx, bool implicit_offsets) {
  NeighborRowAudit audit;
  if (!list || center_idx < 0 || partner_idx < 0 ||
      static_cast<std::size_t>(center_idx) >=
          list->_d_neighbor_num.size() ||
      list->_max_neighbor_capacity <= 0) {
    return audit;
  }

  audit.count = list->_d_neighbor_num[static_cast<std::size_t>(center_idx)];
  audit.written = std::min(std::max(audit.count, rbmd::Id(0)),
                           list->_max_neighbor_capacity);
  std::size_t row_start = 0;
  if (implicit_offsets) {
    row_start = static_cast<std::size_t>(center_idx) *
                static_cast<std::size_t>(list->_max_neighbor_capacity);
  } else {
    if (static_cast<std::size_t>(center_idx) >= list->_start_idx.size()) {
      return audit;
    }
    const rbmd::Id raw_start =
        list->_start_idx[static_cast<std::size_t>(center_idx)];
    if (raw_start < 0) {
      return audit;
    }
    row_start = static_cast<std::size_t>(raw_start);
  }

  const std::size_t row_end =
      row_start + static_cast<std::size_t>(audit.written);
  if (row_end > list->_d_neighbors.size()) {
    return audit;
  }
  const auto begin = list->_d_neighbors.begin() + row_start;
  const auto end = list->_d_neighbors.begin() + row_end;
  audit.contains = thrust::find(begin, end, partner_idx) != end;
  audit.valid = true;
  return audit;
}

void EmitTrackedPairMembershipCsv(
    const std::shared_ptr<DeviceData>& device_data,
    const std::shared_ptr<NeighborList>& neighbor_list,
    const std::shared_ptr<LinkedCell>& linked_cell, const Box& force_box,
    rbmd::Id nlocal) {
  if (!device_data || !neighbor_list || !linked_cell ||
      !ShouldWritePairMembershipDebug(test_current_step)) {
    return;
  }

  constexpr std::array<std::array<rbmd::Id, 2>, 4> tracked_pairs{{
      {{9446364, 9545526}},
      {{9545526, 9446364}},
      {{10446364, 10545526}},
      {{10545526, 10446364}},
  }};
  const rbmd::Id safe_nlocal = std::max<rbmd::Id>(nlocal, 0);
  const rbmd::Id safe_ntotal =
      std::max<rbmd::Id>(linked_cell->_total_atoms_num, 0);
  const auto candidate_list = neighbor_list->_rbl_candidate_list;

  EnsurePairSpikeDebugLogDirectory();
  const int rank = CurrentRank();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/rbl_pair_membership_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,center_gid,partner_gid,center_idx,partner_idx,"
           "center_region,partner_region,distance_sq,center_cell,"
           "partner_cell,candidate_valid,candidate_count,"
           "candidate_written,candidate_contains,core_valid,core_count,"
           "core_written,core_contains\n";
  }

  for (const auto& pair : tracked_pairs) {
    const rbmd::Id center_gid = pair[0];
    const rbmd::Id partner_gid = pair[1];
    const rbmd::Id center_idx = FindGidIndex(
        device_data->_d_atoms_id, safe_ntotal, center_gid);
    const rbmd::Id partner_idx = FindGidIndex(
        device_data->_d_atoms_id, safe_ntotal, partner_gid);
    rbmd::Real distance_sq = rbmd::Real(-1);
    rbmd::Id center_cell = -1;
    rbmd::Id partner_cell = -1;
    if (center_idx >= 0 && partner_idx >= 0) {
      rbmd::Real dx =
          device_data->_d_px[static_cast<std::size_t>(partner_idx)] -
          device_data->_d_px[static_cast<std::size_t>(center_idx)];
      rbmd::Real dy =
          device_data->_d_py[static_cast<std::size_t>(partner_idx)] -
          device_data->_d_py[static_cast<std::size_t>(center_idx)];
      rbmd::Real dz =
          device_data->_d_pz[static_cast<std::size_t>(partner_idx)] -
          device_data->_d_pz[static_cast<std::size_t>(center_idx)];
      MinImageDistance_while(force_box, dx, dy, dz);
      distance_sq = dx * dx + dy * dy + dz * dz;
      if (static_cast<std::size_t>(center_idx) <
          linked_cell->_per_atom_cell_id.size()) {
        center_cell = linked_cell->_per_atom_cell_id[
            static_cast<std::size_t>(center_idx)];
      }
      if (static_cast<std::size_t>(partner_idx) <
          linked_cell->_per_atom_cell_id.size()) {
        partner_cell = linked_cell->_per_atom_cell_id[
            static_cast<std::size_t>(partner_idx)];
      }
    }

    NeighborRowAudit candidate_audit;
    NeighborRowAudit core_audit;
    if (center_idx >= 0 && center_idx < safe_nlocal) {
      candidate_audit = InspectNeighborRow(
          candidate_list, center_idx, partner_idx,
          neighbor_list->_rbl_candidate_implicit_offsets);
      core_audit = InspectNeighborRow(neighbor_list, center_idx, partner_idx,
                                      false);
    }

    csv << test_current_step << "," << rank << "," << center_gid << ","
        << partner_gid << "," << center_idx << "," << partner_idx << ","
        << AtomRegionName(center_idx, safe_nlocal, safe_ntotal) << ","
        << AtomRegionName(partner_idx, safe_nlocal, safe_ntotal) << ","
        << distance_sq << "," << center_cell << "," << partner_cell << ","
        << (candidate_audit.valid ? 1 : 0) << "," << candidate_audit.count
        << "," << candidate_audit.written << ","
        << (candidate_audit.contains ? 1 : 0) << ","
        << (core_audit.valid ? 1 : 0) << "," << core_audit.count << ","
        << core_audit.written << "," << (core_audit.contains ? 1 : 0)
        << "\n";
  }
}

}  // namespace

LJCutCoulKspace::LJCutCoulKspace()
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

LJCutCoulKspace::~LJCutCoulKspace(){}

void LJCutCoulKspace::Init()
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

  _kspace_calculator->Init();
  _alpha = _kspace_calculator->GetAlpha();

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

void LJCutCoulKspace::Execute()
{
  ComputeLJCutCoulForce();
  _kspace_calculator->Execute();
  SumForces();

  EvaluatePotentialEnergy();
}

void LJCutCoulKspace::ComputeLJCutCoulForce()
{
  //
  if ("RBL" ==_neighbor_type)
  {
    ComputeLJRBL();
  }
  else if ("VERLET-SOG" ==_neighbor_type)
  {
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

void LJCutCoulKspace::ComputeLJRBL()
{
    // rbl_neighbor_list_build
    LogLJHipMemory("before_neighbor_build");
    auto start = std::chrono::high_resolution_clock::now();
    _rbl_list = _rbl_neighbor_list_builder->Build();
    LogLJHipMemory("after_neighbor_build");

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<rbmd::Real> duration = end - start;
    TimingStatistics::Instance().record("Neighbor-List",duration.count());
	
    // compute force
	rbmd::revision::WorkflowTimer revision_pair_wall("Pair-Workflow-Wall");
  if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();
  auto start_rbl_force = std::chrono::high_resolution_clock::now();
		
    const auto r_core =
      DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
          "r_core", "hyper_parameters", "neighbor");

     const auto neighbor_sample_num =
     DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
         "neighbor_sample_num", "hyper_parameters", "neighbor");

    auto num_atoms = *(_structure_info_data->_num_atoms);
    const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
    if (_d_rbl_force_sum.size() != 3) {
      _d_rbl_force_sum.resize(3);
    }
    thrust::fill(_d_rbl_force_sum.begin(), _d_rbl_force_sum.end(),
                 rbmd::Real(0));
    const auto rbl_candidate = _rbl_list->_rbl_candidate_list;
    const bool fused_candidate_sampling =
        _rbl_list->_rbl_fused_sampling && static_cast<bool>(rbl_candidate);
    const rbmd::Id* candidate_start = fused_candidate_sampling
        ? thrust::raw_pointer_cast(rbl_candidate->_start_idx.data()) : nullptr;
    const rbmd::Id* candidate_end = fused_candidate_sampling
        ? thrust::raw_pointer_cast(rbl_candidate->_end_idx.data()) : nullptr;
    const rbmd::Id* candidate_num = fused_candidate_sampling
        ? thrust::raw_pointer_cast(rbl_candidate->_d_neighbor_num.data())
        : nullptr;
    const rbmd::Id* candidate_neighbors = fused_candidate_sampling
        ? thrust::raw_pointer_cast(rbl_candidate->_d_neighbors.data())
        : nullptr;
    rbmd::revision::GpuTimer revision_pair("Pair-KernelGPU", nullptr);
    op::LJCutCoulRBLForceOp<device::DEVICE_GPU>()(
        force_box,r_core, _cut_off,num_atoms,neighbor_sample_num,
        _rbl_list->_selection_frequency,_alpha,_qqr2e,
        thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
        thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
        thrust::raw_pointer_cast(_device_data->_d_sigma.data()),
        thrust::raw_pointer_cast(_device_data->_d_eps.data()),
        thrust::raw_pointer_cast(_rbl_list->_start_idx.data()),
        thrust::raw_pointer_cast(_rbl_list->_end_idx.data()),
        thrust::raw_pointer_cast(_rbl_list->_d_neighbors.data()),
        thrust::raw_pointer_cast(_rbl_list->_d_random_neighbor.data()),
        thrust::raw_pointer_cast(_rbl_list->_d_random_neighbor_num.data()),
        thrust::raw_pointer_cast(
            _rbl_list->_d_random_neighbor_weight.data()),
        fused_candidate_sampling, candidate_start, candidate_end,
        candidate_num, candidate_neighbors,
        fused_candidate_sampling ? rbl_candidate->_max_neighbor_capacity : 0,
        _rbl_list->_rbl_candidate_implicit_offsets,
        _rbl_list->_rbl_sample_epoch,
        thrust::raw_pointer_cast(_device_data->_d_charge.data()),
        thrust::raw_pointer_cast(_device_data->_d_px.data()),
        thrust::raw_pointer_cast(_device_data->_d_py.data()),
        thrust::raw_pointer_cast(_device_data->_d_pz.data()),
        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()),
        thrust::raw_pointer_cast(_d_rbl_force_sum.data()));
    revision_pair.Stop();
    const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
    EmitTrackedPairMembershipCsv(_device_data, _rbl_list, linked_cell,
                                 force_box, num_atoms);
    LogLJHipMemory("after_pair_kernel");

    LogLJHipMemory("before_local_reduce");
    CHECK_RUNTIME(MEMCPY(
        _h_rbl_force_sum.data(),
        thrust::raw_pointer_cast(_d_rbl_force_sum.data()),
        3 * sizeof(rbmd::Real), D2H));
    LogLJHipMemory("after_local_reduce");

    std::array<rbmd::Real, 3> global_force_sum{
        _h_rbl_force_sum[0], _h_rbl_force_sum[1], _h_rbl_force_sum[2]};
    AllReduceRealBufferInPlace(global_force_sum.data(),
                               global_force_sum.size());
    LogLJHipMemory("after_mpi_reduce");
    const rbmd::Id global_num_atoms = GetGlobalIdSum(num_atoms);
    _corr_value_x = global_force_sum[0] / global_num_atoms;
    _corr_value_y = global_force_sum[1] / global_num_atoms;
    _corr_value_z = global_force_sum[2] / global_num_atoms;

    // fix RBL:   rbl_force = force - corr_value
    rbmd::revision::GpuTimer revision_correction("RBL-CorrectionGPU", nullptr);
    op::FixRBLForceOp<device::DEVICE_GPU>()(
                         num_atoms, _corr_value_x, _corr_value_y, _corr_value_z,
                        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
                        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
                        thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()));
    revision_correction.Stop();

	if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();

	auto end_rbl_force = std::chrono::high_resolution_clock::now();
	std::chrono::duration<rbmd::Real> duration_rbl_force = end_rbl_force - start_rbl_force;
	TimingStatistics::Instance().record("Short-Range",duration_rbl_force.count());
  
    //energy
   if ("yes" == _energy_rbl_flag){
       ComputeLJCoulEnergy();
    }
}

void LJCutCoulKspace::ComputeLJVerlet()
{
  //neighbor_list_build
  auto start = std::chrono::high_resolution_clock::now();
  _list = _neighbor_list_builder->Build();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record("Neighbor-List",duration.count());
  
  //
  rbmd::revision::WorkflowTimer revision_pair_wall("Pair-Workflow-Wall");
  if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();
  auto start_verlet_force = std::chrono::high_resolution_clock::now();
  
  thrust::device_vector<rbmd::Real> d_total_evdwl(1, 0.0);
  thrust::device_vector<rbmd::Real> d_total_ecoul(1, 0.0);
  //
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  rbmd::revision::GpuTimer revision_pair("Pair-KernelGPU", nullptr);
  op::LJCutCoulForceOp<device::DEVICE_GPU>()(
                    force_box,_cut_off, num_atoms,_alpha,_qqr2e,
                    thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
                    thrust::raw_pointer_cast(_device_data->_d_sigma.data()),
                    thrust::raw_pointer_cast(_device_data->_d_eps.data()),
                    thrust::raw_pointer_cast(_list->_start_idx.data()),
                    thrust::raw_pointer_cast(_list->_end_idx.data()),
                    thrust::raw_pointer_cast(_list->_d_neighbors.data()),
                    thrust::raw_pointer_cast(_device_data->_d_charge.data()),
                    thrust::raw_pointer_cast(_device_data->_d_px.data()),
                    thrust::raw_pointer_cast(_device_data->_d_py.data()),
                    thrust::raw_pointer_cast(_device_data->_d_pz.data()),
                    thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_x.data()),
                    thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_y.data()),
                    thrust::raw_pointer_cast(_device_data->_d_force_ljcoul_z.data()),
                    thrust::raw_pointer_cast(_device_data->_d_flat_virial_lj.data()),
                    thrust::raw_pointer_cast(d_total_evdwl.data()),
                      thrust::raw_pointer_cast(d_total_ecoul.data()));
  revision_pair.Stop();
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  EmitPairSpikeDebugCsv(_device_data, _list, linked_cell, force_box,
                        "after_short_range", _cut_off, _alpha, _qqr2e,
                        num_atoms);

  if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();

  auto end_verlet_force = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration_verlet_force = end_verlet_force - start_verlet_force;
  TimingStatistics::Instance().record("Short-Range",duration_verlet_force.count());
  
  // 
  thrust::host_vector<rbmd::Real> h_total_evdwl(d_total_evdwl);
  thrust::host_vector<rbmd::Real> h_total_ecoul(d_total_ecoul);
  _e_vdwl = GetGlobalRealSum(h_total_evdwl[0]);
  _e_coul = GetGlobalRealSum(h_total_ecoul[0]);
  const rbmd::Id global_num_atoms = GetGlobalIdSum(num_atoms);

  auto unit = DataManager::getInstance().getConfigData()->Get
<std::string>("unit", "init_configuration", "read_data");
  if (ParseUnit(unit) == UNIT::LJ) {
    _e_vdwl = _e_vdwl / global_num_atoms;
    _e_coul = _e_coul / global_num_atoms;
  }

  //sum virial_lj on host
  ReduceVirial(num_atoms,_device_data->_d_flat_virial_lj,
_device_data->_d_virial_lj);

  if (test_current_step ==0 ) {
    thrust::host_vector<rbmd::Real> h_lj_virial =_device_data->_d_virial_lj;
    std::ofstream lj_file("lj_sog_virial.txt");
    if (lj_file.is_open()) {
      for (rbmd::Id i = 0; i < h_lj_virial.size(); ++i) {
        lj_file << i  << " " <<h_lj_virial[i]  << "\n";
      }
      lj_file.close();
    }

  }
}

void LJCutCoulKspace::LJCoulSOGInit() {

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

void LJCutCoulKspace::ComputeLJSOG() {
    //neighbor_list_build
  auto start = std::chrono::high_resolution_clock::now();
  _list = _neighbor_list_builder->Build();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<rbmd::Real> duration = end - start;

  TimingStatistics::Instance().record("Neighbor-List",duration.count());

  //
  if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();
  auto start_verlet_force = std::chrono::high_resolution_clock::now();
  //
  thrust::device_vector<rbmd::Real> _d_total_evdwl(1, 0.0);
  thrust::device_vector<rbmd::Real> _d_total_ecoul(1, 0.0);


  //
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::LJCutCoulForceUserOp<device::DEVICE_GPU>()(
                  force_box, _cut_off, num_atoms,_qqr2e,
                  _rbsog_sigma,_rbsog_b,_rbsog_Mmax,_w0,
                  thrust::raw_pointer_cast(_d_taylor_coeff.data()),
                  thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
                  thrust::raw_pointer_cast(_device_data->_d_sigma.data()),
                  thrust::raw_pointer_cast(_device_data->_d_eps.data()),
                  thrust::raw_pointer_cast(_list->_start_idx.data()),
                  thrust::raw_pointer_cast(_list->_end_idx.data()),
                  thrust::raw_pointer_cast(_list->_d_neighbors.data()),
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

  if (rbmd::revision::GetOptions().isolated_timing) rbmd::revision::DeviceSynchronize();

  auto end_verlet_force = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration_verlet_force = end_verlet_force - start_verlet_force;
  TimingStatistics::Instance().record("Short-Range",duration_verlet_force.count());
  // D2H
  thrust::host_vector<rbmd::Real> h_total_evdwl(_d_total_evdwl);
  thrust::host_vector<rbmd::Real> h_total_ecoul(_d_total_ecoul);
  _e_vdwl = GetGlobalRealSum(h_total_evdwl[0]);
  _e_coul = GetGlobalRealSum(h_total_ecoul[0]);
  const rbmd::Id global_num_atoms = GetGlobalIdSum(num_atoms);

  auto unit = DataManager::getInstance().getConfigData()->Get
<std::string>("unit", "init_configuration", "read_data");
  if (ParseUnit(unit) == UNIT::LJ) {
    _e_vdwl = _e_vdwl / global_num_atoms;
    _e_coul = _e_coul / global_num_atoms;
  }

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


  if (test_current_step ==0 ) {
    thrust::host_vector<rbmd::Real> h_lj_virial =_device_data->_d_virial_lj;
    std::ofstream lj_file("lj_sog_virial.txt");
    if (lj_file.is_open()) {
      for (rbmd::Id i = 0; i < h_lj_virial.size(); ++i) {
        lj_file << i  << " " <<h_lj_virial[i]  << "\n";
      }
      lj_file.close();
    }

  }

}


void LJCutCoulKspace::SumForces()
{
  TransformForces(_device_data->_d_fx,_device_data->_d_force_ljcoul_x,
    _device_data->_d_force_kspace_x);

  TransformForces(_device_data->_d_fy,_device_data->_d_force_ljcoul_y,
    _device_data->_d_force_kspace_y);

  TransformForces(_device_data->_d_fz,_device_data->_d_force_ljcoul_z,
    _device_data->_d_force_kspace_z);
}


void LJCutCoulKspace::ComputeLJCoulEnergy()
{
  // energy
  //neighbor_list_build
  _list = _neighbor_list_builder->Build();

  //
  thrust::device_vector<rbmd::Real> _d_total_evdwl(1, 0.0);
  thrust::device_vector<rbmd::Real> _d_total_ecoul(1, 0.0);
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::LJCutCoulEnergyOp<device::DEVICE_GPU>()(
                force_box,_cut_off,num_atoms,_alpha,_qqr2e,
                thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
                thrust::raw_pointer_cast(_device_data->_d_sigma.data()),
                thrust::raw_pointer_cast(_device_data->_d_eps.data()),
                thrust::raw_pointer_cast(_list->_start_idx.data()),
                thrust::raw_pointer_cast(_list->_end_idx.data()),
                thrust::raw_pointer_cast(_list->_d_neighbors.data()),
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
  const rbmd::Id global_num_atoms = GetGlobalIdSum(num_atoms);

  auto unit = DataManager::getInstance().getConfigData()->Get
<std::string>("unit", "init_configuration", "read_data");
  if (ParseUnit(unit) == UNIT::LJ) {
    _e_vdwl = _e_vdwl / global_num_atoms;
    _e_coul = _e_coul / global_num_atoms;
  }

  //sum virial_lj on host
  ReduceVirial(num_atoms,_device_data->_d_flat_virial_lj,
_device_data->_d_virial_lj);
}

void LJCutCoulKspace::EvaluatePotentialEnergy()
{
  _e_kspace = _kspace_calculator->GetKspacEnergy();
  ThermoStats::Instance().AddThermoData("kspace",_e_kspace);

  _e_pe_rbl = _e_vdwl_rbl + _e_coul_rbl +_e_kspace;
  //test_ave_pe_rbl = _ave_pe_rbl;

  _e_pe = _e_vdwl+ _e_coul +_e_kspace;
  //test_ave_pe = _ave_pe;

  ThermoStats::Instance().AddThermoData("total-potential-energy",_e_pe);
  
  //out
  auto interval = DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
"interval", "outputs", "thermo_out");
  if (!rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    return;
  }

  std::ofstream outfile("thermo.txt", std::ios::app);
  if (outfile.tellp() == 0) {
    outfile << "step e_vdwl e_coul e_kspace e_pe" << std::endl;
  }
  if (test_current_step % interval == 0) {
    outfile << test_current_step << " " << _e_vdwl  << " "<< _e_coul <<" "
      << _e_kspace  << " " << _e_pe<< std::endl;
  }

  outfile.close();
}
