#include "shake_controller.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "data_manager.h"
#include "device_types.h"
#include "common/timing_statistics.hpp"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"
#include "shake_controller_op.h"
#include "unit_factor.h"

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef USE_MPI
#include "rbmd_parallel_until_locator.h"
#endif

extern rbmd::Id test_current_step;

namespace {

using ShakeTimingClock = std::chrono::high_resolution_clock;

void RecordShakeTiming(const char* category,
                       ShakeTimingClock::time_point start) {
  static const bool enabled = [] {
    const char* value = std::getenv("RBMD_DETAILED_SHAKE_TIMING");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
  }();
  if (!enabled) {
    return;
  }
  CHECK_RUNTIME(DEVICESYNC());
  const std::chrono::duration<double> duration =
      ShakeTimingClock::now() - start;
  TimingStatistics::Instance().record(category, duration.count());
}

bool ShakeDebugEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_SHAKE_GUARDS");
  if (!env) {
    return false;
  }
  return env[0] != '\0' && env[0] != '0';
}

bool ShakeDebugPreExchangeEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_SHAKE_PRE_EXCHANGE");
  if (!env) {
    return false;
  }
  return env[0] != '\0' && env[0] != '0';
}

bool ShakeResidualCsvDebugEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_SHAKE_RESIDUAL_CSV");
  if (!env) {
    return false;
  }
  return env[0] != '\0' && std::string(env) != "0";
}

int CurrentShakeProcessId() {
#if defined(_WIN32)
  return static_cast<int>(::_getpid());
#else
  return static_cast<int>(::getpid());
#endif
}

void EnsureShakeResidualDebugLogDirectory() {
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

void EmitShakeDebugLine(const std::string& line) {
  std::fprintf(stderr, "%s\n", line.c_str());
}

std::vector<rbmd::Id> ParseShakeTrackedGidsFromEnv(const char* name) {
  std::vector<rbmd::Id> gids;
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
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

const char* ShakeRegionName(rbmd::Id idx, rbmd::Id native_atoms, rbmd::Id total_atoms) {
  if (idx < 0) {
    return "missing";
  }
  if (idx < native_atoms) {
    return "native";
  }
  if (idx < total_atoms) {
    return "ghost";
  }
  return "out_of_total";
}

Box GetShakeBoxForMpi(const Box& local_box) {
#ifdef USE_MPI
  if (!GET_RBMD_PARALLEL) {
    return local_box;
  }
  return GET_RBMD_PARALLEL->_global_structure_info.global_box;
#else
  return local_box;
#endif
}

Box GetShakeCorrectionBoxForMpi(const Box& local_box) {
#ifdef USE_MPI
  if (!GET_RBMD_PARALLEL) {
    return local_box;
  }

  Box correction_box = GET_RBMD_PARALLEL->_global_structure_info.global_box;
  if (GET_RBMD_PARALLEL->_domdec) {
    const auto& grid = GET_RBMD_PARALLEL->_domdec->_grid_size;
    if (grid[0] > 1) {
      correction_box._pbc_x = false;
    }
    if (grid[1] > 1) {
      correction_box._pbc_y = false;
    }
    if (grid[2] > 1) {
      correction_box._pbc_z = false;
    }
  }
  return correction_box;
#else
  return local_box;
#endif
}

rbmd::Real HostMinImageDelta(rbmd::Real delta, rbmd::Real box_length, bool pbc_enabled) {
  if (!pbc_enabled || box_length <= rbmd::Real(0)) {
    return delta;
  }
  return delta - std::nearbyint(delta / box_length) * box_length;
}

rbmd::Real HostDistanceSquared(rbmd::Real x0, rbmd::Real y0, rbmd::Real z0,
                               rbmd::Real x1, rbmd::Real y1, rbmd::Real z1,
                               const Box& box) {
  const rbmd::Real dx =
      HostMinImageDelta(x0 - x1, box._length[0], box._pbc_x);
  const rbmd::Real dy =
      HostMinImageDelta(y0 - y1, box._length[1], box._pbc_y);
  const rbmd::Real dz =
      HostMinImageDelta(z0 - z1, box._length[2], box._pbc_z);
  return dx * dx + dy * dy + dz * dz;
}

rbmd::Real RawDistanceSquared(const thrust::host_vector<rbmd::Real>& h_px,
                              const thrust::host_vector<rbmd::Real>& h_py,
                              const thrust::host_vector<rbmd::Real>& h_pz,
                              rbmd::Id lhs_idx, rbmd::Id rhs_idx) {
  const std::size_t lhs = static_cast<std::size_t>(lhs_idx);
  const std::size_t rhs = static_cast<std::size_t>(rhs_idx);
  const rbmd::Real dx = h_px[lhs] - h_px[rhs];
  const rbmd::Real dy = h_py[lhs] - h_py[rhs];
  const rbmd::Real dz = h_pz[lhs] - h_pz[rhs];
  return dx * dx + dy * dy + dz * dz;
}

#ifdef USE_MPI

Box GetGlobalPbcBoxForGhostImage(const Box& fallback_box) {
  if (!GET_RBMD_PARALLEL) {
    return fallback_box;
  }
  return GET_RBMD_PARALLEL->_global_structure_info.global_box;
}

rbmd::Real InferGhostPeriodicShiftDim(rbmd::Real ghost_value,
                                      rbmd::Real owner_value,
                                      rbmd::Real box_length,
                                      bool pbc_enabled) {
  if (!pbc_enabled || box_length <= rbmd::Real(0)) {
    return rbmd::Real(0);
  }

  const rbmd::Real raw_delta = ghost_value - owner_value;
  const long long image_count =
      static_cast<long long>(std::llround(raw_delta / box_length));
  return static_cast<rbmd::Real>(image_count) * box_length;
}

struct ShakePeriodicShift {
  rbmd::Real x = 0;
  rbmd::Real y = 0;
  rbmd::Real z = 0;
};

ShakePeriodicShift InferGhostPeriodicShift(rbmd::Real ghost_px,
                                           rbmd::Real ghost_py,
                                           rbmd::Real ghost_pz,
                                           rbmd::Real owner_px,
                                           rbmd::Real owner_py,
                                           rbmd::Real owner_pz,
                                           const Box& global_pbc_box) {
  ShakePeriodicShift shift;
  shift.x = InferGhostPeriodicShiftDim(ghost_px, owner_px,
                                       global_pbc_box._length[0],
                                       global_pbc_box._pbc_x);
  shift.y = InferGhostPeriodicShiftDim(ghost_py, owner_py,
                                       global_pbc_box._length[1],
                                       global_pbc_box._pbc_y);
  shift.z = InferGhostPeriodicShiftDim(ghost_pz, owner_pz,
                                       global_pbc_box._length[2],
                                       global_pbc_box._pbc_z);
  return shift;
}

struct ShakeForwardRecord {
  rbmd::Id gid = -1;
  rbmd::Real px = 0;
  rbmd::Real py = 0;
  rbmd::Real pz = 0;
  rbmd::Real shake_px = 0;
  rbmd::Real shake_py = 0;
  rbmd::Real shake_pz = 0;
  rbmd::Real vx = 0;
  rbmd::Real vy = 0;
  rbmd::Real vz = 0;
  rbmd::Real shake_vx = 0;
  rbmd::Real shake_vy = 0;
  rbmd::Real shake_vz = 0;
};

struct ShakeReverseRecord {
  rbmd::Id gid = -1;
  rbmd::Real dx = 0;
  rbmd::Real dy = 0;
  rbmd::Real dz = 0;
  rbmd::Real dvx = 0;
  rbmd::Real dvy = 0;
  rbmd::Real dvz = 0;
};

struct ShakeGhostFetchRecord {
  rbmd::Id gid = -1;
  rbmd::Id type = -1;
  rbmd::Real px = 0;
  rbmd::Real py = 0;
  rbmd::Real pz = 0;
  rbmd::Real shake_px = 0;
  rbmd::Real shake_py = 0;
  rbmd::Real shake_pz = 0;
  rbmd::Real vx = 0;
  rbmd::Real vy = 0;
  rbmd::Real vz = 0;
  rbmd::Real shake_vx = 0;
  rbmd::Real shake_vy = 0;
  rbmd::Real shake_vz = 0;
};

struct ShakeGhostReference {
  rbmd::Real px = 0;
  rbmd::Real py = 0;
  rbmd::Real pz = 0;
  rbmd::Real shake_px = 0;
  rbmd::Real shake_py = 0;
  rbmd::Real shake_pz = 0;
};

template <typename T>
void ExchangeByOwner(const std::vector<int>& send_counts,
                     const std::vector<T>& send_data,
                     std::vector<int>& recv_counts,
                     std::vector<T>& recv_data, MPI_Comm comm) {
  const int nranks = static_cast<int>(send_counts.size());
  recv_counts.assign(nranks, 0);
  MPI_CHECK(MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                         MPI_INT, comm));

  std::vector<int> send_bytes(nranks, 0);
  std::vector<int> recv_bytes(nranks, 0);
  std::vector<int> send_displs(nranks, 0);
  std::vector<int> recv_displs(nranks, 0);
  int total_send_bytes = 0;
  int total_recv_bytes = 0;
  for (int rank = 0; rank < nranks; ++rank) {
    send_bytes[rank] = send_counts[rank] * static_cast<int>(sizeof(T));
    recv_bytes[rank] = recv_counts[rank] * static_cast<int>(sizeof(T));
    send_displs[rank] = total_send_bytes;
    recv_displs[rank] = total_recv_bytes;
    total_send_bytes += send_bytes[rank];
    total_recv_bytes += recv_bytes[rank];
  }

  recv_data.resize(static_cast<std::size_t>(total_recv_bytes / sizeof(T)));
  MPI_CHECK(MPI_Alltoallv(
      send_data.empty() ? nullptr : reinterpret_cast<const char*>(send_data.data()),
      send_bytes.data(), send_displs.data(), MPI_BYTE,
      recv_data.empty() ? nullptr : reinterpret_cast<char*>(recv_data.data()),
      recv_bytes.data(), recv_displs.data(), MPI_BYTE, comm));
}

bool HasAnyShakeCorrection(const ShakeReverseRecord& record) {
  return record.dx != rbmd::Real(0) || record.dy != rbmd::Real(0) ||
         record.dz != rbmd::Real(0) || record.dvx != rbmd::Real(0) ||
         record.dvy != rbmd::Real(0) || record.dvz != rbmd::Real(0);
}

rbmd::Id SelectClosestShakeReplicaIndex(
    rbmd::Id gid, rbmd::Id anchor_idx, rbmd::Id native_atoms,
    const std::unordered_map<rbmd::Id, std::vector<rbmd::Id>>& indices_by_gid,
    const thrust::host_vector<rbmd::Real>& h_ref_px,
    const thrust::host_vector<rbmd::Real>& h_ref_py,
    const thrust::host_vector<rbmd::Real>& h_ref_pz,
    const Box& distance_box) {
  const auto iter = indices_by_gid.find(gid);
  if (iter == indices_by_gid.end() || iter->second.empty()) {
    return rbmd::Id(-1);
  }

  for (rbmd::Id candidate_idx : iter->second) {
    if (candidate_idx >= 0 && candidate_idx < native_atoms) {
      return candidate_idx;
    }
  }

  if (anchor_idx < 0) {
    return iter->second.front();
  }

  rbmd::Id best_idx = -1;
  rbmd::Real best_dist2 = std::numeric_limits<rbmd::Real>::max();
  for (rbmd::Id candidate_idx : iter->second) {
    const std::size_t candidate = static_cast<std::size_t>(candidate_idx);
    const std::size_t anchor = static_cast<std::size_t>(anchor_idx);
    const rbmd::Real dist2 = HostDistanceSquared(
        h_ref_px[candidate], h_ref_py[candidate], h_ref_pz[candidate],
        h_ref_px[anchor], h_ref_py[anchor], h_ref_pz[anchor], distance_box);
    const bool prefer_candidate =
        dist2 < best_dist2 ||
        (dist2 == best_dist2 && candidate_idx < native_atoms &&
         (best_idx < 0 || best_idx >= native_atoms));
    if (prefer_candidate) {
      best_dist2 = dist2;
      best_idx = candidate_idx;
    }
  }
  return best_idx;
}

rbmd::Id SelectOwnedShakeAnchorIndex(
    rbmd::Id gid, rbmd::Id native_atoms,
    const std::unordered_map<rbmd::Id, std::vector<rbmd::Id>>& indices_by_gid) {
  const auto iter = indices_by_gid.find(gid);
  if (iter == indices_by_gid.end() || iter->second.empty()) {
    return rbmd::Id(-1);
  }
  for (rbmd::Id candidate_idx : iter->second) {
    if (candidate_idx >= 0 && candidate_idx < native_atoms) {
      return candidate_idx;
    }
  }
  return iter->second.front();
}

std::unordered_map<rbmd::Id, int> BuildNativeOwnerRankMap(
    const thrust::host_vector<rbmd::Id>& h_ids, rbmd::Id native_atoms,
    int total_ranks, MPI_Comm comm) {
  const int local_count = static_cast<int>(native_atoms);
  std::vector<int> counts(static_cast<std::size_t>(total_ranks), 0);
  MPI_CHECK(MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm));

  std::vector<int> counts_bytes(static_cast<std::size_t>(total_ranks), 0);
  std::vector<int> displs_bytes(static_cast<std::size_t>(total_ranks), 0);
  int total_ids = 0;
  int total_bytes = 0;
  for (int rank = 0; rank < total_ranks; ++rank) {
    counts_bytes[static_cast<std::size_t>(rank)] =
        counts[static_cast<std::size_t>(rank)] * static_cast<int>(sizeof(rbmd::Id));
    displs_bytes[static_cast<std::size_t>(rank)] = total_bytes;
    total_ids += counts[static_cast<std::size_t>(rank)];
    total_bytes += counts_bytes[static_cast<std::size_t>(rank)];
  }

  std::vector<rbmd::Id> all_native_ids(static_cast<std::size_t>(total_ids));
  MPI_CHECK(MPI_Allgatherv(
      native_atoms > 0 ? reinterpret_cast<const char*>(h_ids.data()) : nullptr,
      local_count * static_cast<int>(sizeof(rbmd::Id)), MPI_BYTE,
      all_native_ids.empty() ? nullptr : reinterpret_cast<char*>(all_native_ids.data()),
      counts_bytes.data(), displs_bytes.data(), MPI_BYTE, comm));

  std::unordered_map<rbmd::Id, int> owner_rank_by_gid;
  owner_rank_by_gid.reserve(all_native_ids.size());
  std::size_t offset = 0;
  for (int rank = 0; rank < total_ranks; ++rank) {
    const int rank_count = counts[static_cast<std::size_t>(rank)];
    for (int i = 0; i < rank_count; ++i) {
      owner_rank_by_gid.emplace(all_native_ids[offset + static_cast<std::size_t>(i)], rank);
    }
    offset += static_cast<std::size_t>(rank_count);
  }
  return owner_rank_by_gid;
}

rbmd::Real InferTopologyFetchShiftDim(rbmd::Real reference_value,
                                      rbmd::Real owner_value,
                                      rbmd::Real box_length,
                                      bool pbc_enabled) {
  if (!pbc_enabled || box_length <= rbmd::Real(0)) {
    return rbmd::Real(0);
  }

  const rbmd::Real desired_delta = reference_value - owner_value;
  const long long image_count =
      static_cast<long long>(std::llround(desired_delta / box_length));
  return static_cast<rbmd::Real>(image_count) * box_length;
}

ShakePeriodicShift InferTopologyFetchShift(rbmd::Real reference_px,
                                           rbmd::Real reference_py,
                                           rbmd::Real reference_pz,
                                           rbmd::Real owner_px,
                                           rbmd::Real owner_py,
                                           rbmd::Real owner_pz,
                                           const Box& global_pbc_box) {
  ShakePeriodicShift shift;
  shift.x = InferTopologyFetchShiftDim(reference_px, owner_px,
                                       global_pbc_box._length[0],
                                       global_pbc_box._pbc_x);
  shift.y = InferTopologyFetchShiftDim(reference_py, owner_py,
                                       global_pbc_box._length[1],
                                       global_pbc_box._pbc_y);
  shift.z = InferTopologyFetchShiftDim(reference_pz, owner_pz,
                                       global_pbc_box._length[2],
                                       global_pbc_box._pbc_z);
  return shift;
}

std::vector<rbmd::Id> CollectMissingShakeClusterGhostGids(
    const thrust::host_vector<rbmd::Id>& owner_atom0,
    const thrust::host_vector<rbmd::Id>& owner_atom1,
    const thrust::host_vector<rbmd::Id>& owner_atom2,
    const std::unordered_set<rbmd::Id>& present_ids) {
  std::vector<rbmd::Id> missing_cluster_gids;
  missing_cluster_gids.reserve(owner_atom0.size());
  for (std::size_t i = 0; i < owner_atom0.size(); ++i) {
    const rbmd::Id atom0 = owner_atom0[i];
    const rbmd::Id atom1 = owner_atom1[i];
    const rbmd::Id atom2 = owner_atom2[i];
    if (present_ids.find(atom0) == present_ids.end()) {
      missing_cluster_gids.push_back(atom0);
    }
    if (present_ids.find(atom1) == present_ids.end()) {
      missing_cluster_gids.push_back(atom1);
    }
    if (present_ids.find(atom2) == present_ids.end()) {
      missing_cluster_gids.push_back(atom2);
    }
  }
  std::sort(missing_cluster_gids.begin(), missing_cluster_gids.end());
  missing_cluster_gids.erase(
      std::unique(missing_cluster_gids.begin(), missing_cluster_gids.end()),
      missing_cluster_gids.end());
  return missing_cluster_gids;
}

std::unordered_map<rbmd::Id, ShakeGhostReference> BuildMissingShakeGhostReferences(
    const thrust::host_vector<rbmd::Id>& owner_atom0,
    const thrust::host_vector<rbmd::Id>& owner_atom1,
    const thrust::host_vector<rbmd::Id>& owner_atom2,
    const thrust::host_vector<rbmd::Id>& h_all_ids,
    const thrust::host_vector<rbmd::Real>& h_px,
    const thrust::host_vector<rbmd::Real>& h_py,
    const thrust::host_vector<rbmd::Real>& h_pz,
    const thrust::host_vector<rbmd::Real>& h_shake_px,
    const thrust::host_vector<rbmd::Real>& h_shake_py,
    const thrust::host_vector<rbmd::Real>& h_shake_pz,
    const std::unordered_set<rbmd::Id>& present_ids) {
  std::unordered_map<rbmd::Id, rbmd::Id> local_idx_by_gid;
  local_idx_by_gid.reserve(h_all_ids.size());
  for (rbmd::Id atom_idx = 0; atom_idx < static_cast<rbmd::Id>(h_all_ids.size());
       ++atom_idx) {
    local_idx_by_gid.emplace(h_all_ids[static_cast<std::size_t>(atom_idx)], atom_idx);
  }

  auto make_reference = [&](rbmd::Id local_idx) {
    const std::size_t idx = static_cast<std::size_t>(local_idx);
    ShakeGhostReference reference;
    reference.px = h_px[idx];
    reference.py = h_py[idx];
    reference.pz = h_pz[idx];
    reference.shake_px = h_shake_px[idx];
    reference.shake_py = h_shake_py[idx];
    reference.shake_pz = h_shake_pz[idx];
    return reference;
  };

  std::unordered_map<rbmd::Id, ShakeGhostReference> reference_by_gid;
  reference_by_gid.reserve(owner_atom0.size());
  for (std::size_t i = 0; i < owner_atom0.size(); ++i) {
    const rbmd::Id atom0 = owner_atom0[i];
    const rbmd::Id atom1 = owner_atom1[i];
    const rbmd::Id atom2 = owner_atom2[i];
    const bool missing0 = present_ids.find(atom0) == present_ids.end();
    const bool missing1 = present_ids.find(atom1) == present_ids.end();
    const bool missing2 = present_ids.find(atom2) == present_ids.end();
    if (!missing0 && !missing1 && !missing2) {
      continue;
    }

    rbmd::Id reference_idx = -1;
    const rbmd::Id candidates[3] = {atom1, atom0, atom2};
    for (rbmd::Id candidate_gid : candidates) {
      const auto candidate_iter = local_idx_by_gid.find(candidate_gid);
      if (candidate_iter == local_idx_by_gid.end()) {
        continue;
      }
      reference_idx = candidate_iter->second;
      break;
    }
    if (reference_idx < 0) {
      continue;
    }

    const ShakeGhostReference reference = make_reference(reference_idx);
    if (missing0) {
      reference_by_gid.emplace(atom0, reference);
    }
    if (missing1) {
      reference_by_gid.emplace(atom1, reference);
    }
    if (missing2) {
      reference_by_gid.emplace(atom2, reference);
    }
  }

  return reference_by_gid;
}

void RefreshShakeGhostAtomIdToIdxMap(LinkedCell* linked_cell,
                                     const std::shared_ptr<DeviceData>& device_data) {
  if (!linked_cell || !device_data) {
    return;
  }

  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  if (total_atoms < 0) {
    return;
  }

  thrust::host_vector<rbmd::Id> h_ids(
      device_data->_d_atoms_id.begin(),
      device_data->_d_atoms_id.begin() + total_atoms);
  thrust::host_vector<rbmd::Id> h_atom_id_to_idx(linked_cell->_atom_id_to_idx.size(),
                                                 rbmd::Id(-1));
  const std::size_t map_size = h_atom_id_to_idx.size();
  for (rbmd::Id atom_idx = 0; atom_idx < total_atoms; ++atom_idx) {
    const rbmd::Id gid = h_ids[static_cast<std::size_t>(atom_idx)];
    if (gid < 0 || static_cast<std::size_t>(gid) >= map_size) {
      continue;
    }
    h_atom_id_to_idx[static_cast<std::size_t>(gid)] = atom_idx;
  }
  thrust::copy(h_atom_id_to_idx.begin(), h_atom_id_to_idx.end(),
               linked_cell->_atom_id_to_idx.begin());
}

void AppendFetchedShakeGhostAtoms(
    LinkedCell* linked_cell, const std::shared_ptr<DeviceData>& device_data,
    const std::vector<ShakeGhostFetchRecord>& fetched_records,
    const std::unordered_map<rbmd::Id, ShakeGhostReference>& reference_by_gid,
    const Box& global_pbc_box) {
  if (!linked_cell || !device_data || fetched_records.empty()) {
    return;
  }

  thrust::host_vector<rbmd::Id> h_existing_ids(
      device_data->_d_atoms_id.begin(),
      device_data->_d_atoms_id.begin() + linked_cell->_total_atoms_num);
  std::unordered_set<rbmd::Id> present_ids;
  present_ids.reserve(h_existing_ids.size());
  for (rbmd::Id gid : h_existing_ids) {
    present_ids.insert(gid);
  }

  std::vector<ShakeGhostFetchRecord> append_records;
  append_records.reserve(fetched_records.size());
  for (const auto& record : fetched_records) {
    if (record.gid < 0 || record.type < 0) {
      continue;
    }
    if (present_ids.find(record.gid) != present_ids.end()) {
      continue;
    }
    append_records.push_back(record);
    present_ids.insert(record.gid);
  }

  if (append_records.empty()) {
    return;
  }

  const rbmd::Id old_total_atoms = linked_cell->_total_atoms_num;
  const rbmd::Id append_count = static_cast<rbmd::Id>(append_records.size());
  linked_cell->EnsureCapacity(old_total_atoms + append_count);
  linked_cell->UpdateGhostNum(append_count);

  thrust::host_vector<rbmd::Id> h_append_ids(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Id> h_append_types(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_px(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_py(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_pz(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_shake_px(
      static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_shake_py(
      static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_shake_pz(
      static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_vx(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_vy(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_vz(static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_shake_vx(
      static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_shake_vy(
      static_cast<std::size_t>(append_count));
  thrust::host_vector<rbmd::Real> h_append_shake_vz(
      static_cast<std::size_t>(append_count));

  for (std::size_t i = 0; i < append_records.size(); ++i) {
    const auto& record = append_records[i];
    h_append_ids[i] = record.gid;
    h_append_types[i] = record.type;
    h_append_vx[i] = record.vx;
    h_append_vy[i] = record.vy;
    h_append_vz[i] = record.vz;
    h_append_shake_vx[i] = record.shake_vx;
    h_append_shake_vy[i] = record.shake_vy;
    h_append_shake_vz[i] = record.shake_vz;

    ShakePeriodicShift current_shift;
    ShakePeriodicShift shadow_shift;
    const auto ref_iter = reference_by_gid.find(record.gid);
    if (ref_iter != reference_by_gid.end()) {
      current_shift = InferTopologyFetchShift(
          ref_iter->second.px, ref_iter->second.py, ref_iter->second.pz, record.px,
          record.py, record.pz, global_pbc_box);
      shadow_shift = InferTopologyFetchShift(
          ref_iter->second.shake_px, ref_iter->second.shake_py,
          ref_iter->second.shake_pz, record.shake_px, record.shake_py,
          record.shake_pz, global_pbc_box);
    }

    h_append_px[i] = record.px + current_shift.x;
    h_append_py[i] = record.py + current_shift.y;
    h_append_pz[i] = record.pz + current_shift.z;
    h_append_shake_px[i] = record.shake_px + shadow_shift.x;
    h_append_shake_py[i] = record.shake_py + shadow_shift.y;
    h_append_shake_pz[i] = record.shake_pz + shadow_shift.z;
  }

  thrust::copy(h_append_ids.begin(), h_append_ids.end(),
               device_data->_d_atoms_id.begin() + old_total_atoms);
  thrust::copy(h_append_types.begin(), h_append_types.end(),
               device_data->_d_atoms_type.begin() + old_total_atoms);
  thrust::copy(h_append_px.begin(), h_append_px.end(),
               device_data->_d_px.begin() + old_total_atoms);
  thrust::copy(h_append_py.begin(), h_append_py.end(),
               device_data->_d_py.begin() + old_total_atoms);
  thrust::copy(h_append_pz.begin(), h_append_pz.end(),
               device_data->_d_pz.begin() + old_total_atoms);
  thrust::copy(h_append_shake_px.begin(), h_append_shake_px.end(),
               device_data->_d_shake_px.begin() + old_total_atoms);
  thrust::copy(h_append_shake_py.begin(), h_append_shake_py.end(),
               device_data->_d_shake_py.begin() + old_total_atoms);
  thrust::copy(h_append_shake_pz.begin(), h_append_shake_pz.end(),
               device_data->_d_shake_pz.begin() + old_total_atoms);
  thrust::copy(h_append_vx.begin(), h_append_vx.end(),
               device_data->_d_vx.begin() + old_total_atoms);
  thrust::copy(h_append_vy.begin(), h_append_vy.end(),
               device_data->_d_vy.begin() + old_total_atoms);
  thrust::copy(h_append_vz.begin(), h_append_vz.end(),
               device_data->_d_vz.begin() + old_total_atoms);
  thrust::copy(h_append_shake_vx.begin(), h_append_shake_vx.end(),
               device_data->_d_shake_vx.begin() + old_total_atoms);
  thrust::copy(h_append_shake_vy.begin(), h_append_shake_vy.end(),
               device_data->_d_shake_vy.begin() + old_total_atoms);
  thrust::copy(h_append_shake_vz.begin(), h_append_shake_vz.end(),
               device_data->_d_shake_vz.begin() + old_total_atoms);

  RefreshShakeGhostAtomIdToIdxMap(linked_cell, device_data);

  if (ShakeDebugEnabled()) {
    std::ostringstream oss;
    oss << "[shake-debug] appended topology-fetched ghost atoms=" << append_count;
    if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
      oss << " rank=" << GET_RBMD_PARALLEL->_domdec->_current_rank
          << "/" << GET_RBMD_PARALLEL->_domdec->_total_ranks;
    }
    EmitShakeDebugLine(oss.str());
  }
}

void FetchMissingShakeGhostAtomsForMpi(
    LinkedCell* linked_cell, const std::shared_ptr<DeviceData>& device_data,
    const std::shared_ptr<Box>& box,
    const std::vector<rbmd::Id>& missing_cluster_gids,
    const thrust::host_vector<rbmd::Id>& owner_atom0,
    const thrust::host_vector<rbmd::Id>& owner_atom1,
    const thrust::host_vector<rbmd::Id>& owner_atom2,
    const std::unordered_map<rbmd::Id, int>& owner_rank_by_gid) {
  if (!linked_cell || !device_data || !box) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  const int current_rank = GET_RBMD_PARALLEL->_domdec->_current_rank;
  const int total_ranks = GET_RBMD_PARALLEL->_domdec->_total_ranks;
  auto mpi_comm = GET_RBMD_PARALLEL->_domdec->_mpi_comm;

  thrust::host_vector<rbmd::Id> h_all_ids(device_data->_d_atoms_id.begin(),
                                          device_data->_d_atoms_id.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_all_px(device_data->_d_px.begin(),
                                           device_data->_d_px.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_all_py(device_data->_d_py.begin(),
                                           device_data->_d_py.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_all_pz(device_data->_d_pz.begin(),
                                           device_data->_d_pz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_all_shake_px(
      device_data->_d_shake_px.begin(), device_data->_d_shake_px.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_all_shake_py(
      device_data->_d_shake_py.begin(), device_data->_d_shake_py.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_all_shake_pz(
      device_data->_d_shake_pz.begin(), device_data->_d_shake_pz.begin() + total_atoms);

  std::unordered_set<rbmd::Id> present_ids;
  present_ids.reserve(h_all_ids.size());
  for (rbmd::Id gid : h_all_ids) {
    present_ids.insert(gid);
  }

  const auto reference_by_gid = BuildMissingShakeGhostReferences(
      owner_atom0, owner_atom1, owner_atom2, h_all_ids, h_all_px, h_all_py, h_all_pz,
      h_all_shake_px, h_all_shake_py, h_all_shake_pz, present_ids);

  std::vector<std::vector<rbmd::Id>> request_ids_by_rank(
      static_cast<std::size_t>(total_ranks));
  for (rbmd::Id gid : missing_cluster_gids) {
    const auto owner_iter = owner_rank_by_gid.find(gid);
    if (owner_iter == owner_rank_by_gid.end()) {
      continue;
    }
    const int owner_rank = owner_iter->second;
    if (owner_rank < 0 || owner_rank >= total_ranks || owner_rank == current_rank) {
      continue;
    }
    request_ids_by_rank[static_cast<std::size_t>(owner_rank)].push_back(gid);
  }

  std::vector<int> send_counts(static_cast<std::size_t>(total_ranks), 0);
  std::vector<rbmd::Id> send_requests;
  send_requests.reserve(missing_cluster_gids.size());
  for (int rank = 0; rank < total_ranks; ++rank) {
    const auto& request_ids =
        request_ids_by_rank[static_cast<std::size_t>(rank)];
    send_counts[static_cast<std::size_t>(rank)] =
        static_cast<int>(request_ids.size());
    send_requests.insert(send_requests.end(),
                         request_ids_by_rank[static_cast<std::size_t>(rank)].begin(),
                         request_ids_by_rank[static_cast<std::size_t>(rank)].end());
  }

  std::vector<int> recv_counts;
  std::vector<rbmd::Id> recv_requests;
  ExchangeByOwner(send_counts, send_requests, recv_counts, recv_requests, mpi_comm);

  thrust::host_vector<rbmd::Id> h_native_ids(device_data->_d_atoms_id.begin(),
                                             device_data->_d_atoms_id.begin() + native_atoms);
  thrust::host_vector<rbmd::Id> h_native_types(
      device_data->_d_atoms_type.begin(), device_data->_d_atoms_type.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_px(device_data->_d_px.begin(),
                                              device_data->_d_px.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_py(device_data->_d_py.begin(),
                                              device_data->_d_py.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_pz(device_data->_d_pz.begin(),
                                              device_data->_d_pz.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_shake_px(
      device_data->_d_shake_px.begin(), device_data->_d_shake_px.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_shake_py(
      device_data->_d_shake_py.begin(), device_data->_d_shake_py.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_shake_pz(
      device_data->_d_shake_pz.begin(), device_data->_d_shake_pz.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_vx(device_data->_d_vx.begin(),
                                              device_data->_d_vx.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_vy(device_data->_d_vy.begin(),
                                              device_data->_d_vy.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_vz(device_data->_d_vz.begin(),
                                              device_data->_d_vz.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_shake_vx(
      device_data->_d_shake_vx.begin(), device_data->_d_shake_vx.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_shake_vy(
      device_data->_d_shake_vy.begin(), device_data->_d_shake_vy.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_native_shake_vz(
      device_data->_d_shake_vz.begin(), device_data->_d_shake_vz.begin() + native_atoms);

  std::unordered_map<rbmd::Id, rbmd::Id> native_idx_by_gid;
  native_idx_by_gid.reserve(static_cast<std::size_t>(native_atoms));
  for (rbmd::Id atom_idx = 0; atom_idx < native_atoms; ++atom_idx) {
    native_idx_by_gid.emplace(h_native_ids[static_cast<std::size_t>(atom_idx)], atom_idx);
  }

  std::vector<ShakeGhostFetchRecord> send_payloads;
  send_payloads.reserve(recv_requests.size());
  std::vector<rbmd::Id> missing_owner_payload_gids;
  missing_owner_payload_gids.reserve(recv_requests.size());
  for (rbmd::Id gid : recv_requests) {
    const auto native_iter = native_idx_by_gid.find(gid);
    if (native_iter == native_idx_by_gid.end()) {
      missing_owner_payload_gids.push_back(gid);
      continue;
    }
    ShakeGhostFetchRecord record;
    const rbmd::Id native_idx = native_iter->second;
    const std::size_t idx = static_cast<std::size_t>(native_idx);
    record.gid = gid;
    record.type = h_native_types[idx];
    record.px = h_native_px[idx];
    record.py = h_native_py[idx];
    record.pz = h_native_pz[idx];
    record.shake_px = h_native_shake_px[idx];
    record.shake_py = h_native_shake_py[idx];
    record.shake_pz = h_native_shake_pz[idx];
    record.vx = h_native_vx[idx];
    record.vy = h_native_vy[idx];
    record.vz = h_native_vz[idx];
    record.shake_vx = h_native_shake_vx[idx];
    record.shake_vy = h_native_shake_vy[idx];
    record.shake_vz = h_native_shake_vz[idx];
    send_payloads.push_back(record);
  }

  if (!missing_owner_payload_gids.empty()) {
    std::sort(missing_owner_payload_gids.begin(), missing_owner_payload_gids.end());
    missing_owner_payload_gids.erase(
        std::unique(missing_owner_payload_gids.begin(),
                    missing_owner_payload_gids.end()),
        missing_owner_payload_gids.end());
    std::ostringstream oss;
    oss << "MPI SHAKE topology fetch owner missing requested native gids on rank "
        << current_rank << ": ";
    for (std::size_t i = 0; i < missing_owner_payload_gids.size(); ++i) {
      if (i != 0) {
        oss << ",";
      }
      oss << missing_owner_payload_gids[i];
    }
    throw std::runtime_error(oss.str());
  }

  std::vector<int> payload_recv_counts;
  std::vector<ShakeGhostFetchRecord> recv_payloads;
  ExchangeByOwner(recv_counts, send_payloads, payload_recv_counts, recv_payloads,
                  mpi_comm);

  AppendFetchedShakeGhostAtoms(linked_cell, device_data, recv_payloads,
                               reference_by_gid,
                               GetGlobalPbcBoxForGhostImage(*box));
}

void EmitShakeTrackedPresence(
    const std::string& phase,
    const std::vector<rbmd::Id>& gids,
    const thrust::host_vector<rbmd::Id>& h_ids,
    rbmd::Id native_atoms,
    rbmd::Id total_atoms,
    int current_rank,
    const std::unordered_map<rbmd::Id, int>& owner_rank_by_gid) {
  if (!ShakeDebugEnabled() || gids.empty()) {
    return;
  }

  std::unordered_map<rbmd::Id, rbmd::Id> local_idx_by_gid;
  local_idx_by_gid.reserve(static_cast<std::size_t>(total_atoms));
  for (rbmd::Id atom_idx = 0; atom_idx < total_atoms; ++atom_idx) {
    local_idx_by_gid.emplace(h_ids[static_cast<std::size_t>(atom_idx)], atom_idx);
  }

  std::ostringstream oss;
  oss << "[shake-debug] rank " << current_rank
      << " tracked gid presence phase=" << phase << ": ";
  bool first = true;
  for (rbmd::Id gid : gids) {
    if (!first) {
      oss << "; ";
    }
    first = false;

    rbmd::Id local_idx = -1;
    const auto local_iter = local_idx_by_gid.find(gid);
    if (local_iter != local_idx_by_gid.end()) {
      local_idx = local_iter->second;
    }

    int owner_rank = -1;
    const auto owner_iter = owner_rank_by_gid.find(gid);
    if (owner_iter != owner_rank_by_gid.end()) {
      owner_rank = owner_iter->second;
    }

    oss << "gid=" << gid
        << " owner_rank=" << owner_rank
        << " local_idx=" << local_idx
        << " region=" << ShakeRegionName(local_idx, native_atoms, total_atoms);
  }
  EmitShakeDebugLine(oss.str());
}

#endif

}  // namespace

ShakeController::ShakeController()
    : _structure_info_data(
          DataManager::getInstance().getMDData()->_structure_info_data),
      _device_data(DataManager::getInstance().getDeviceData()),
      _box(DataManager::getInstance().getMDData()->_box) {}

void ShakeController::Init() {
  _dt = DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
      "timestep", "execution");
  auto unit = DataManager::getInstance().getConfigData()->Get<std::string>(
      "unit", "init_configuration", "read_data");
  UNIT unit_factor = ParseUnit(unit);

  switch (unit_factor) {
    case UNIT::METAL:
      _fmt2v = UnitFactor<UNIT::METAL>::_fmt2v;
      break;
    case UNIT::LJ:
      _fmt2v = UnitFactor<UNIT::LJ>::_fmt2v;
      break;
    case UNIT::REAL:
      _fmt2v = UnitFactor<UNIT::REAL>::_fmt2v;
      break;
    default:
      break;
  }
}

bool ShakeController::UseMpiOwnedShakePath() const {
#ifdef USE_MPI
  if (!GET_RBMD_PARALLEL || !GET_RBMD_PARALLEL->_domdec) {
    return false;
  }
  return GET_RBMD_PARALLEL->_domdec->_total_ranks > 1;
#else
  return false;
#endif
}

void ShakeController::BuildOwnedShakeClustersForMpi() {
  _device_data->_d_shake_owner_atom0.clear();
  _device_data->_d_shake_owner_atom1.clear();
  _device_data->_d_shake_owner_atom2.clear();
  _device_data->_d_shake_owner_anchor.clear();
  _device_data->_d_shake_owner_idx0.clear();
  _device_data->_d_shake_owner_idx1.clear();
  _device_data->_d_shake_owner_idx2.clear();

#ifndef USE_MPI
  return;
#else
  if (!UseMpiOwnedShakePath()) {
    return;
  }

  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const int angle_per_atom = _device_data->angle_per_atom;
  const std::size_t angle_slots =
      native_atoms > 0 && angle_per_atom > 0
          ? static_cast<std::size_t>(native_atoms) *
                static_cast<std::size_t>(angle_per_atom)
          : 0;
  const bool local_topology_invalid =
      angle_per_atom <= 0 || native_atoms < 0 ||
      _device_data->d_num_angle.size() < static_cast<std::size_t>(native_atoms) ||
      _device_data->d_angle_atom1.size() < angle_slots ||
      _device_data->d_angle_atom2.size() < angle_slots ||
      _device_data->d_angle_atom3.size() < angle_slots;
  int local_invalid = local_topology_invalid ? 1 : 0;
  int global_invalid = 0;
  MPI_CHECK(MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT, MPI_MAX,
                          GET_RBMD_PARALLEL->_domdec->_mpi_comm));
  if (global_invalid != 0) {
    throw std::runtime_error(
        "MPI SHAKE topology buffers are incomplete on at least one rank");
  }

  thrust::host_vector<rbmd::Id> h_atoms_id(
      _device_data->_d_atoms_id.begin(),
      _device_data->_d_atoms_id.begin() + native_atoms);
  thrust::host_vector<rbmd::Id> h_all_ids(
      _device_data->_d_atoms_id.begin(),
      _device_data->_d_atoms_id.begin() + linked_cell->_total_atoms_num);
  thrust::host_vector<int> h_num_angle(
      _device_data->d_num_angle.begin(),
      _device_data->d_num_angle.begin() + native_atoms);

  thrust::host_vector<rbmd::Id> h_angle_atom1(
      _device_data->d_angle_atom1.begin(),
      _device_data->d_angle_atom1.begin() + angle_slots);
  thrust::host_vector<rbmd::Id> h_angle_atom2(
      _device_data->d_angle_atom2.begin(),
      _device_data->d_angle_atom2.begin() + angle_slots);
  thrust::host_vector<rbmd::Id> h_angle_atom3(
      _device_data->d_angle_atom3.begin(),
      _device_data->d_angle_atom3.begin() + angle_slots);

  thrust::host_vector<rbmd::Id> owner_atom0;
  thrust::host_vector<rbmd::Id> owner_atom1;
  thrust::host_vector<rbmd::Id> owner_atom2;
  thrust::host_vector<rbmd::Id> owner_anchor;
  owner_atom0.reserve(static_cast<std::size_t>(native_atoms));
  owner_atom1.reserve(static_cast<std::size_t>(native_atoms));
  owner_atom2.reserve(static_cast<std::size_t>(native_atoms));
  owner_anchor.reserve(static_cast<std::size_t>(native_atoms));

  for (rbmd::Id atom_idx = 0; atom_idx < native_atoms; ++atom_idx) {
    const rbmd::Id owned_gid = h_atoms_id[static_cast<std::size_t>(atom_idx)];
    const int count =
        std::min(std::max(h_num_angle[static_cast<std::size_t>(atom_idx)], 0),
                 angle_per_atom);
    for (int slot = 0; slot < count; ++slot) {
      const std::size_t flat =
          static_cast<std::size_t>(atom_idx) * static_cast<std::size_t>(angle_per_atom) +
          static_cast<std::size_t>(slot);
      const rbmd::Id atom0 = h_angle_atom1[flat];
      const rbmd::Id atom1 = h_angle_atom2[flat];
      const rbmd::Id atom2 = h_angle_atom3[flat];
      if (atom0 < 0 || atom1 < 0 || atom2 < 0) {
        continue;
      }
      if (atom1 != owned_gid) {
        continue;
      }
      owner_atom0.push_back(atom0);
      owner_atom1.push_back(atom1);
      owner_atom2.push_back(atom2);
      owner_anchor.push_back(owned_gid);
    }
  }

  std::unordered_set<rbmd::Id> present_ids;
  present_ids.reserve(h_all_ids.size());
  for (rbmd::Id gid : h_all_ids) {
    present_ids.insert(gid);
  }

  int current_rank = -1;
#ifdef USE_MPI
  if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
    current_rank = GET_RBMD_PARALLEL->_domdec->_current_rank;
  }
#endif
  const int total_ranks = GET_RBMD_PARALLEL->_domdec->_total_ranks;
  auto mpi_comm = GET_RBMD_PARALLEL->_domdec->_mpi_comm;
  const auto owner_rank_by_gid =
      BuildNativeOwnerRankMap(h_all_ids, native_atoms, total_ranks, mpi_comm);
  auto tracked_gids = ParseShakeTrackedGidsFromEnv("RBMD_DEBUG_TRACK_GIDS");

  const auto missing_cluster_gids = CollectMissingShakeClusterGhostGids(
      owner_atom0, owner_atom1, owner_atom2, present_ids);
  FetchMissingShakeGhostAtomsForMpi(linked_cell.get(), _device_data, _box,
                                    missing_cluster_gids, owner_atom0,
                                    owner_atom1, owner_atom2, owner_rank_by_gid);
  if (!missing_cluster_gids.empty()) {
    h_all_ids = thrust::host_vector<rbmd::Id>(
        _device_data->_d_atoms_id.begin(),
        _device_data->_d_atoms_id.begin() + linked_cell->_total_atoms_num);
    present_ids.clear();
    present_ids.reserve(h_all_ids.size());
    for (rbmd::Id gid : h_all_ids) {
      present_ids.insert(gid);
    }
  }

  std::ostringstream missing_cluster_message;
  int missing_cluster_count = 0;
  std::vector<rbmd::Id> missing_debug_gids = tracked_gids;
  for (std::size_t i = 0; i < owner_anchor.size(); ++i) {
    const rbmd::Id atom0 = owner_atom0[i];
    const rbmd::Id atom1 = owner_atom1[i];
    const rbmd::Id atom2 = owner_atom2[i];
    const bool missing0 = present_ids.find(atom0) == present_ids.end();
    const bool missing1 = present_ids.find(atom1) == present_ids.end();
    const bool missing2 = present_ids.find(atom2) == present_ids.end();
    if (!missing0 && !missing1 && !missing2) {
      continue;
    }

    missing_debug_gids.push_back(atom0);
    missing_debug_gids.push_back(atom1);
    missing_debug_gids.push_back(atom2);

    if (missing_cluster_count == 0) {
      missing_cluster_message
          << "MPI SHAKE owner cluster missing local/ghost atoms on rank "
          << current_rank << ": ";
    } else {
      missing_cluster_message << "; ";
    }
    missing_cluster_message << "anchor=" << owner_anchor[i]
                            << " cluster=(" << atom0 << "," << atom1 << ","
                            << atom2 << ")"
                            << " missing=[" << (missing0 ? "1" : "0") << ","
                            << (missing1 ? "1" : "0") << ","
                            << (missing2 ? "1" : "0") << "]";
    ++missing_cluster_count;
    if (missing_cluster_count >= 8) {
      break;
    }
  }

  if (missing_cluster_count > 0) {
    std::sort(missing_debug_gids.begin(), missing_debug_gids.end());
    missing_debug_gids.erase(
        std::unique(missing_debug_gids.begin(), missing_debug_gids.end()),
        missing_debug_gids.end());
    EmitShakeTrackedPresence("build_owner_cluster_missing", missing_debug_gids,
                             h_all_ids, native_atoms, linked_cell->_total_atoms_num,
                             current_rank, owner_rank_by_gid);
    throw std::runtime_error(missing_cluster_message.str());
  }

  if (ShakeDebugEnabled()) {
    int current_rank = -1;
#ifdef USE_MPI
    if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
      current_rank = GET_RBMD_PARALLEL->_domdec->_current_rank;
    }
#endif

    int duplicate_cluster_count = 0;
    std::ostringstream duplicate_message;
    for (std::size_t i = 0; i < owner_anchor.size(); ++i) {
      const rbmd::Id atom0 = owner_atom0[i];
      const rbmd::Id atom1 = owner_atom1[i];
      const rbmd::Id atom2 = owner_atom2[i];
      if (atom0 != atom1 && atom1 != atom2 && atom0 != atom2) {
        continue;
      }
      if (duplicate_cluster_count == 0) {
        duplicate_message << "[shake-debug] rank " << current_rank
                          << " duplicate owner cluster atoms: ";
      } else {
        duplicate_message << "; ";
      }
      duplicate_message << "anchor=" << owner_anchor[i] << " cluster=(" << atom0
                        << "," << atom1 << "," << atom2 << ")";
      ++duplicate_cluster_count;
      if (duplicate_cluster_count >= 8) {
        break;
      }
    }

    std::ostringstream summary;
    summary << "[shake-debug] rank " << current_rank
            << " owner clusters=" << owner_anchor.size()
            << " native_atoms=" << native_atoms
            << " angle_per_atom=" << angle_per_atom;
    if (duplicate_cluster_count > 0) {
      summary << " duplicate_clusters=" << duplicate_cluster_count;
    }
    EmitShakeDebugLine(summary.str());
    if (duplicate_cluster_count > 0) {
      EmitShakeDebugLine(duplicate_message.str());
    }
  }

  _device_data->_d_shake_owner_atom0 = owner_atom0;
  _device_data->_d_shake_owner_atom1 = owner_atom1;
  _device_data->_d_shake_owner_atom2 = owner_atom2;
  _device_data->_d_shake_owner_anchor = owner_anchor;
#endif
}

void ShakeController::ResolveOwnedShakeClusterIndicesForMpi(
    ShakeCommStage stage) {
#ifdef USE_MPI
  if (!UseMpiOwnedShakePath()) {
    return;
  }

  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  if (_device_data->_d_shake_owner_anchor.empty() || total_atoms <= 0) {
    _device_data->_d_shake_owner_idx0.clear();
    _device_data->_d_shake_owner_idx1.clear();
    _device_data->_d_shake_owner_idx2.clear();
    return;
  }

  thrust::host_vector<rbmd::Id> h_ids(_device_data->_d_atoms_id.begin(),
                                      _device_data->_d_atoms_id.begin() + total_atoms);
  thrust::host_vector<rbmd::Id> h_owner_atom0(
      _device_data->_d_shake_owner_atom0.begin(),
      _device_data->_d_shake_owner_atom0.end());
  thrust::host_vector<rbmd::Id> h_owner_atom1(
      _device_data->_d_shake_owner_atom1.begin(),
      _device_data->_d_shake_owner_atom1.end());
  thrust::host_vector<rbmd::Id> h_owner_atom2(
      _device_data->_d_shake_owner_atom2.begin(),
      _device_data->_d_shake_owner_atom2.end());
  thrust::host_vector<rbmd::Id> h_owner_anchor(
      _device_data->_d_shake_owner_anchor.begin(),
      _device_data->_d_shake_owner_anchor.end());

  const bool use_shadow_position =
      stage == ShakeCommStage::PredictedPositionForward;
  const auto& d_ref_px =
      use_shadow_position ? _device_data->_d_shake_px : _device_data->_d_px;
  const auto& d_ref_py =
      use_shadow_position ? _device_data->_d_shake_py : _device_data->_d_py;
  const auto& d_ref_pz =
      use_shadow_position ? _device_data->_d_shake_pz : _device_data->_d_pz;
  const Box distance_box = GetGlobalPbcBoxForGhostImage(*_box);
  thrust::host_vector<rbmd::Real> h_ref_px(
      d_ref_px.begin(), d_ref_px.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_ref_py(
      d_ref_py.begin(), d_ref_py.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_ref_pz(
      d_ref_pz.begin(), d_ref_pz.begin() + total_atoms);

  std::unordered_map<rbmd::Id, std::vector<rbmd::Id>> indices_by_gid;
  indices_by_gid.reserve(static_cast<std::size_t>(total_atoms));
  for (rbmd::Id atom_idx = 0; atom_idx < total_atoms; ++atom_idx) {
    indices_by_gid[h_ids[static_cast<std::size_t>(atom_idx)]].push_back(atom_idx);
  }

  thrust::host_vector<rbmd::Id> h_owner_idx0(h_owner_anchor.size(), rbmd::Id(-1));
  thrust::host_vector<rbmd::Id> h_owner_idx1(h_owner_anchor.size(), rbmd::Id(-1));
  thrust::host_vector<rbmd::Id> h_owner_idx2(h_owner_anchor.size(), rbmd::Id(-1));

  for (std::size_t i = 0; i < h_owner_anchor.size(); ++i) {
    const rbmd::Id anchor_gid = h_owner_anchor[i];
    const rbmd::Id anchor_idx =
        SelectOwnedShakeAnchorIndex(anchor_gid, native_atoms, indices_by_gid);
    const rbmd::Id idx0 = SelectClosestShakeReplicaIndex(
        h_owner_atom0[i], anchor_idx, native_atoms, indices_by_gid, h_ref_px,
        h_ref_py, h_ref_pz, distance_box);
    const rbmd::Id idx2 = SelectClosestShakeReplicaIndex(
        h_owner_atom2[i], anchor_idx, native_atoms, indices_by_gid, h_ref_px,
        h_ref_py, h_ref_pz, distance_box);
    if (anchor_idx < 0 || idx0 < 0 || idx2 < 0) {
      std::ostringstream oss;
      oss << "MPI SHAKE failed to resolve owner cluster replica index: anchor_gid="
          << anchor_gid << " cluster=(" << h_owner_atom0[i] << ","
          << h_owner_atom1[i] << "," << h_owner_atom2[i] << ")"
          << " resolved=(" << idx0 << "," << anchor_idx << "," << idx2 << ")";
      throw std::runtime_error(oss.str());
    }
    h_owner_idx0[i] = idx0;
    h_owner_idx1[i] = anchor_idx;
    h_owner_idx2[i] = idx2;
  }

  _device_data->_d_shake_owner_idx0 = h_owner_idx0;
  _device_data->_d_shake_owner_idx1 = h_owner_idx1;
  _device_data->_d_shake_owner_idx2 = h_owner_idx2;
#else
  (void)stage;
#endif
}

void ShakeController::ForwardSyncReplicatedShakeState(ShakeCommStage stage) {
#ifdef USE_MPI
  if (!UseMpiOwnedShakePath()) {
    return;
  }

  ForwardGhostState forward_state;
  switch (stage) {
    case ShakeCommStage::PredictedPositionForward:
      forward_state = ForwardGhostState::PredictedPositionForward;
      break;
    case ShakeCommStage::CorrectedStateForward:
      forward_state = ForwardGhostState::CorrectedStateForward;
      break;
    case ShakeCommStage::VelocityForward:
      forward_state = ForwardGhostState::VelocityForward;
      break;
    case ShakeCommStage::VelocityCorrectedForward:
      forward_state = ForwardGhostState::VelocityCorrectedForward;
      break;
    default:
      throw std::invalid_argument(
          "Unsupported SHAKE stage for forward ghost exchange");
  }
  RbmdParallelUntilLocator::GetInstance()
      .GetRbmdParallelUntil()
      ->ForwardExchangeGhostState(forward_state);
#else
  (void)stage;
#endif
}

void ShakeController::ForwardSyncShakeState(ShakeCommStage stage) {
#ifdef USE_MPI
  if (!UseMpiOwnedShakePath()) {
    return;
  }

  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  const int current_rank = GET_RBMD_PARALLEL->_domdec->_current_rank;
  const int total_ranks = GET_RBMD_PARALLEL->_domdec->_total_ranks;
  auto mpi_comm = GET_RBMD_PARALLEL->_domdec->_mpi_comm;

  thrust::host_vector<rbmd::Id> h_ids(_device_data->_d_atoms_id.begin(),
                                      _device_data->_d_atoms_id.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_px(_device_data->_d_px.begin(),
                                       _device_data->_d_px.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_py(_device_data->_d_py.begin(),
                                       _device_data->_d_py.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_pz(_device_data->_d_pz.begin(),
                                       _device_data->_d_pz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_vx(_device_data->_d_vx.begin(),
                                       _device_data->_d_vx.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_vy(_device_data->_d_vy.begin(),
                                       _device_data->_d_vy.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_vz(_device_data->_d_vz.begin(),
                                       _device_data->_d_vz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_shake_px(
      _device_data->_d_shake_px.begin(),
      _device_data->_d_shake_px.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_shake_py(
      _device_data->_d_shake_py.begin(),
      _device_data->_d_shake_py.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_shake_pz(
      _device_data->_d_shake_pz.begin(),
      _device_data->_d_shake_pz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_shake_vx(
      _device_data->_d_shake_vx.begin(),
      _device_data->_d_shake_vx.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_shake_vy(
      _device_data->_d_shake_vy.begin(),
      _device_data->_d_shake_vy.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_shake_vz(
      _device_data->_d_shake_vz.begin(),
      _device_data->_d_shake_vz.begin() + total_atoms);

  const bool sync_position =
      stage == ShakeCommStage::PredictedPositionForward ||
      stage == ShakeCommStage::CorrectedStateForward;
  const bool sync_velocity =
      stage == ShakeCommStage::PredictedPositionForward ||
      stage == ShakeCommStage::VelocityForward ||
      stage == ShakeCommStage::VelocityCorrectedForward ||
      stage == ShakeCommStage::CorrectedStateForward;
  const bool sync_shadow_position =
      stage == ShakeCommStage::PredictedPositionForward ||
      stage == ShakeCommStage::CorrectedStateForward;
  const bool sync_shadow_velocity =
      stage == ShakeCommStage::CorrectedStateForward ||
      stage == ShakeCommStage::VelocityCorrectedForward;
  const Box global_pbc_box = GetGlobalPbcBoxForGhostImage(*_box);

  const auto owner_rank_by_gid =
      BuildNativeOwnerRankMap(h_ids, native_atoms, total_ranks, mpi_comm);

  std::vector<std::vector<rbmd::Id>> request_ids_by_rank(
      static_cast<std::size_t>(total_ranks));
  std::unordered_map<rbmd::Id, std::vector<rbmd::Id>> ghost_indices_by_gid;
  for (rbmd::Id atom_idx = native_atoms; atom_idx < total_atoms; ++atom_idx) {
    const std::size_t idx = static_cast<std::size_t>(atom_idx);
    const auto owner_iter = owner_rank_by_gid.find(h_ids[idx]);
    if (owner_iter == owner_rank_by_gid.end()) {
      continue;
    }
    const int owner_rank = owner_iter->second;
    if (owner_rank == current_rank) {
      continue;
    }
    request_ids_by_rank[static_cast<std::size_t>(owner_rank)].push_back(h_ids[idx]);
    ghost_indices_by_gid[h_ids[idx]].push_back(atom_idx);
  }

  std::vector<int> send_counts(static_cast<std::size_t>(total_ranks), 0);
  std::vector<rbmd::Id> send_requests;
  for (int rank = 0; rank < total_ranks; ++rank) {
    send_counts[static_cast<std::size_t>(rank)] =
        static_cast<int>(request_ids_by_rank[static_cast<std::size_t>(rank)].size());
    send_requests.insert(send_requests.end(),
                         request_ids_by_rank[static_cast<std::size_t>(rank)].begin(),
                         request_ids_by_rank[static_cast<std::size_t>(rank)].end());
  }

  std::vector<int> recv_counts;
  std::vector<rbmd::Id> recv_requests;
  ExchangeByOwner(send_counts, send_requests, recv_counts, recv_requests, mpi_comm);

  std::unordered_map<rbmd::Id, rbmd::Id> native_idx_by_gid;
  native_idx_by_gid.reserve(static_cast<std::size_t>(native_atoms));
  for (rbmd::Id atom_idx = 0; atom_idx < native_atoms; ++atom_idx) {
    native_idx_by_gid.emplace(h_ids[static_cast<std::size_t>(atom_idx)], atom_idx);
  }

  std::vector<ShakeForwardRecord> send_payloads;
  send_payloads.reserve(recv_requests.size());
  for (rbmd::Id gid : recv_requests) {
    ShakeForwardRecord record;
    record.gid = -1;
    const auto iter = native_idx_by_gid.find(gid);
    if (iter != native_idx_by_gid.end()) {
      const std::size_t native_idx = static_cast<std::size_t>(iter->second);
      record.gid = gid;
      if (sync_position) {
        record.px = h_px[native_idx];
        record.py = h_py[native_idx];
        record.pz = h_pz[native_idx];
      }
      if (sync_shadow_position) {
        record.shake_px = h_shake_px[native_idx];
        record.shake_py = h_shake_py[native_idx];
        record.shake_pz = h_shake_pz[native_idx];
      }
      if (sync_velocity) {
        record.vx = h_vx[native_idx];
        record.vy = h_vy[native_idx];
        record.vz = h_vz[native_idx];
      }
      if (sync_shadow_velocity) {
        record.shake_vx = h_shake_vx[native_idx];
        record.shake_vy = h_shake_vy[native_idx];
        record.shake_vz = h_shake_vz[native_idx];
      }
    }
    send_payloads.push_back(record);
  }

  std::vector<ShakeForwardRecord> recv_payloads;
  std::vector<int> payload_recv_counts;
  ExchangeByOwner(recv_counts, send_payloads, payload_recv_counts, recv_payloads,
                  mpi_comm);

  for (const auto& record : recv_payloads) {
    if (record.gid < 0) {
      continue;
    }
    const auto ghost_iter = ghost_indices_by_gid.find(record.gid);
    if (ghost_iter == ghost_indices_by_gid.end()) {
      continue;
    }
    for (rbmd::Id atom_idx : ghost_iter->second) {
      const std::size_t idx = static_cast<std::size_t>(atom_idx);
      // Keep each ghost replica on the same periodic image it already occupies,
      // mirroring LAMMPS forward_comm() semantics without rebuilding sendlists here.
      ShakePeriodicShift shadow_shift{};
      if (sync_shadow_position) {
        shadow_shift = InferGhostPeriodicShift(
            h_shake_px[idx], h_shake_py[idx], h_shake_pz[idx], record.shake_px,
            record.shake_py, record.shake_pz, global_pbc_box);
      }
      ShakePeriodicShift current_shift{};
      if (sync_position) {
        current_shift = InferGhostPeriodicShift(
            h_px[idx], h_py[idx], h_pz[idx], record.px, record.py, record.pz,
            global_pbc_box);
      }

      if (sync_position) {
        h_px[idx] = record.px + current_shift.x;
        h_py[idx] = record.py + current_shift.y;
        h_pz[idx] = record.pz + current_shift.z;
      }
      if (sync_shadow_position) {
        const ShakePeriodicShift slot_shift =
            sync_position ? current_shift : shadow_shift;
        h_shake_px[idx] = record.shake_px + slot_shift.x;
        h_shake_py[idx] = record.shake_py + slot_shift.y;
        h_shake_pz[idx] = record.shake_pz + slot_shift.z;
      }
      if (sync_velocity) {
        h_vx[idx] = record.vx;
        h_vy[idx] = record.vy;
        h_vz[idx] = record.vz;
      }
      if (sync_shadow_velocity) {
        h_shake_vx[idx] = record.shake_vx;
        h_shake_vy[idx] = record.shake_vy;
        h_shake_vz[idx] = record.shake_vz;
      }
    }
  }

  if (sync_position) {
    thrust::copy(h_px.begin() + native_atoms, h_px.begin() + total_atoms,
                 _device_data->_d_px.begin() + native_atoms);
    thrust::copy(h_py.begin() + native_atoms, h_py.begin() + total_atoms,
                 _device_data->_d_py.begin() + native_atoms);
    thrust::copy(h_pz.begin() + native_atoms, h_pz.begin() + total_atoms,
                 _device_data->_d_pz.begin() + native_atoms);
  }
  if (sync_velocity) {
    thrust::copy(h_vx.begin() + native_atoms, h_vx.begin() + total_atoms,
                 _device_data->_d_vx.begin() + native_atoms);
    thrust::copy(h_vy.begin() + native_atoms, h_vy.begin() + total_atoms,
                 _device_data->_d_vy.begin() + native_atoms);
    thrust::copy(h_vz.begin() + native_atoms, h_vz.begin() + total_atoms,
                 _device_data->_d_vz.begin() + native_atoms);
  }
  if (sync_shadow_position) {
    thrust::copy(h_shake_px.begin() + native_atoms, h_shake_px.begin() + total_atoms,
                 _device_data->_d_shake_px.begin() + native_atoms);
    thrust::copy(h_shake_py.begin() + native_atoms, h_shake_py.begin() + total_atoms,
                 _device_data->_d_shake_py.begin() + native_atoms);
    thrust::copy(h_shake_pz.begin() + native_atoms, h_shake_pz.begin() + total_atoms,
                 _device_data->_d_shake_pz.begin() + native_atoms);
  }
  if (sync_shadow_velocity) {
    thrust::copy(h_shake_vx.begin() + native_atoms, h_shake_vx.begin() + total_atoms,
                 _device_data->_d_shake_vx.begin() + native_atoms);
    thrust::copy(h_shake_vy.begin() + native_atoms, h_shake_vy.begin() + total_atoms,
                 _device_data->_d_shake_vy.begin() + native_atoms);
    thrust::copy(h_shake_vz.begin() + native_atoms, h_shake_vz.begin() + total_atoms,
                 _device_data->_d_shake_vz.begin() + native_atoms);
  }
#else
  (void)stage;
#endif
}

void ShakeController::ReverseAccumulateShakeCorrections(ShakeCommStage stage) {
#ifdef USE_MPI
  if (!UseMpiOwnedShakePath()) {
    return;
  }
  if (stage != ShakeCommStage::CorrectionReverse) {
    return;
  }

  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  const int current_rank = GET_RBMD_PARALLEL->_domdec->_current_rank;
  const int total_ranks = GET_RBMD_PARALLEL->_domdec->_total_ranks;
  auto mpi_comm = GET_RBMD_PARALLEL->_domdec->_mpi_comm;

  thrust::host_vector<rbmd::Id> h_ids(_device_data->_d_atoms_id.begin(),
                                      _device_data->_d_atoms_id.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_px(_device_data->_d_px.begin(),
                                       _device_data->_d_px.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_py(_device_data->_d_py.begin(),
                                       _device_data->_d_py.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_pz(_device_data->_d_pz.begin(),
                                       _device_data->_d_pz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_dx(_device_data->_d_shake_dx.begin(),
                                       _device_data->_d_shake_dx.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_dy(_device_data->_d_shake_dy.begin(),
                                       _device_data->_d_shake_dy.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_dz(_device_data->_d_shake_dz.begin(),
                                       _device_data->_d_shake_dz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_dvx(_device_data->_d_shake_dvx.begin(),
                                        _device_data->_d_shake_dvx.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_dvy(_device_data->_d_shake_dvy.begin(),
                                        _device_data->_d_shake_dvy.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_dvz(_device_data->_d_shake_dvz.begin(),
                                        _device_data->_d_shake_dvz.begin() + total_atoms);

  const auto owner_rank_by_gid =
      BuildNativeOwnerRankMap(h_ids, native_atoms, total_ranks, mpi_comm);

  std::vector<std::vector<ShakeReverseRecord>> send_records_by_rank(
      static_cast<std::size_t>(total_ranks));
  for (rbmd::Id atom_idx = native_atoms; atom_idx < total_atoms; ++atom_idx) {
    const std::size_t idx = static_cast<std::size_t>(atom_idx);
    ShakeReverseRecord record;
    record.gid = h_ids[idx];
    record.dx = h_dx[idx];
    record.dy = h_dy[idx];
    record.dz = h_dz[idx];
    record.dvx = h_dvx[idx];
    record.dvy = h_dvy[idx];
    record.dvz = h_dvz[idx];
    if (!HasAnyShakeCorrection(record)) {
      continue;
    }
    const auto owner_iter = owner_rank_by_gid.find(record.gid);
    if (owner_iter == owner_rank_by_gid.end()) {
      continue;
    }
    const int owner_rank = owner_iter->second;
    if (owner_rank == current_rank) {
      continue;
    }
    send_records_by_rank[static_cast<std::size_t>(owner_rank)].push_back(record);
  }

  std::vector<int> send_counts(static_cast<std::size_t>(total_ranks), 0);
  std::vector<ShakeReverseRecord> send_payloads;
  for (int rank = 0; rank < total_ranks; ++rank) {
    send_counts[static_cast<std::size_t>(rank)] = static_cast<int>(
        send_records_by_rank[static_cast<std::size_t>(rank)].size());
    send_payloads.insert(send_payloads.end(),
                         send_records_by_rank[static_cast<std::size_t>(rank)].begin(),
                         send_records_by_rank[static_cast<std::size_t>(rank)].end());
  }

  std::vector<int> recv_counts;
  std::vector<ShakeReverseRecord> recv_payloads;
  ExchangeByOwner(send_counts, send_payloads, recv_counts, recv_payloads, mpi_comm);

  std::unordered_map<rbmd::Id, rbmd::Id> native_idx_by_gid;
  native_idx_by_gid.reserve(static_cast<std::size_t>(native_atoms));
  for (rbmd::Id atom_idx = 0; atom_idx < native_atoms; ++atom_idx) {
    native_idx_by_gid.emplace(h_ids[static_cast<std::size_t>(atom_idx)], atom_idx);
  }

  for (const auto& record : recv_payloads) {
    const auto iter = native_idx_by_gid.find(record.gid);
    if (iter == native_idx_by_gid.end()) {
      continue;
    }
    const std::size_t idx = static_cast<std::size_t>(iter->second);
    h_dx[idx] += record.dx;
    h_dy[idx] += record.dy;
    h_dz[idx] += record.dz;
    h_dvx[idx] += record.dvx;
    h_dvy[idx] += record.dvy;
    h_dvz[idx] += record.dvz;
  }

  thrust::copy(h_dx.begin(), h_dx.begin() + native_atoms, _device_data->_d_shake_dx.begin());
  thrust::copy(h_dy.begin(), h_dy.begin() + native_atoms, _device_data->_d_shake_dy.begin());
  thrust::copy(h_dz.begin(), h_dz.begin() + native_atoms, _device_data->_d_shake_dz.begin());
  thrust::copy(h_dvx.begin(), h_dvx.begin() + native_atoms,
               _device_data->_d_shake_dvx.begin());
  thrust::copy(h_dvy.begin(), h_dvy.begin() + native_atoms,
               _device_data->_d_shake_dvy.begin());
  thrust::copy(h_dvz.begin(), h_dvz.begin() + native_atoms,
               _device_data->_d_shake_dvz.begin());
#else
  (void)stage;
#endif
}

void ShakeController::EnsureReplicatedShakeClusters() {
#ifndef USE_MPI
  return;
#else
  if (!UseMpiOwnedShakePath()) {
    return;
  }

  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  auto parallel =
      RbmdParallelUntilLocator::GetInstance().GetRbmdParallelUntil();
  if (!linked_cell || !parallel) {
    throw std::runtime_error(
        "MPI SHAKE cannot build clusters without halo state");
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  const int angle_per_atom = _device_data->angle_per_atom;
  const std::uint32_t epoch = parallel->ForwardExchangeEpoch();
  if (_shake_cluster_epoch == epoch &&
      _shake_cluster_native_atoms == native_atoms &&
      _shake_cluster_total_atoms == total_atoms &&
      _shake_cluster_angle_per_atom == angle_per_atom) {
    return;
  }

  _device_data->_d_shake_cluster_idx0.clear();
  _device_data->_d_shake_cluster_idx1.clear();
  _device_data->_d_shake_cluster_idx2.clear();

  if (native_atoms < 0 || total_atoms < native_atoms) {
    throw std::runtime_error("MPI SHAKE observed invalid native/total counts");
  }
  if (native_atoms == 0) {
    _shake_cluster_epoch = epoch;
    _shake_cluster_native_atoms = native_atoms;
    _shake_cluster_total_atoms = total_atoms;
    _shake_cluster_angle_per_atom = angle_per_atom;
    return;
  }
  if (angle_per_atom <= 0) {
    throw std::runtime_error(
        "MPI SHAKE requires positive per-atom angle capacity");
  }

  const std::size_t native_size = static_cast<std::size_t>(native_atoms);
  const std::size_t total_size = static_cast<std::size_t>(total_atoms);
  if (_device_data->_d_atoms_id.size() < total_size ||
      _device_data->_d_px.size() < total_size ||
      _device_data->_d_py.size() < total_size ||
      _device_data->_d_pz.size() < total_size ||
      _device_data->d_num_angle.size() < native_size) {
    throw std::runtime_error(
        "MPI SHAKE atom buffers are incomplete for the current halo epoch");
  }
  if (native_size >
      std::numeric_limits<std::size_t>::max() /
          static_cast<std::size_t>(angle_per_atom)) {
    throw std::overflow_error("MPI SHAKE angle slot count overflow");
  }
  const std::size_t angle_slots =
      native_size * static_cast<std::size_t>(angle_per_atom);
  if (_device_data->d_angle_atom1.size() < angle_slots ||
      _device_data->d_angle_atom2.size() < angle_slots ||
      _device_data->d_angle_atom3.size() < angle_slots) {
    throw std::runtime_error(
        "MPI SHAKE per-atom angle buffers are incomplete");
  }
  if (angle_slots >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "MPI SHAKE local cluster candidate count exceeds int capacity");
  }

  _device_data->_d_shake_sorted_gid.assign(
      _device_data->_d_atoms_id.begin(),
      _device_data->_d_atoms_id.begin() + total_atoms);
  _device_data->_d_shake_sorted_idx.resize(total_size);
  thrust::sequence(_device_data->_d_shake_sorted_idx.begin(),
                   _device_data->_d_shake_sorted_idx.end(), rbmd::Id(0));
  thrust::sort_by_key(_device_data->_d_shake_sorted_gid.begin(),
                      _device_data->_d_shake_sorted_gid.end(),
                      _device_data->_d_shake_sorted_idx.begin());

  _device_data->_d_shake_cluster_idx0.resize(angle_slots);
  _device_data->_d_shake_cluster_idx1.resize(angle_slots);
  _device_data->_d_shake_cluster_idx2.resize(angle_slots);
  thrust::device_vector<int> d_cluster_count(1, 0);
  thrust::device_vector<int> d_invalid_count(1, 0);
  thrust::device_vector<rbmd::Id> d_first_invalid_gids(3, rbmd::Id(-1));

  op::BuildShakeClustersOp<device::DEVICE_GPU>()(
      native_atoms, total_atoms, angle_per_atom,
      thrust::raw_pointer_cast(_device_data->d_num_angle.data()),
      thrust::raw_pointer_cast(_device_data->d_angle_atom1.data()),
      thrust::raw_pointer_cast(_device_data->d_angle_atom2.data()),
      thrust::raw_pointer_cast(_device_data->d_angle_atom3.data()),
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_sorted_gid.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_sorted_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx0.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx1.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx2.data()),
      thrust::raw_pointer_cast(d_cluster_count.data()),
      thrust::raw_pointer_cast(d_invalid_count.data()),
      thrust::raw_pointer_cast(d_first_invalid_gids.data()));

  const int cluster_count = d_cluster_count[0];
  const int invalid_count = d_invalid_count[0];
  if (invalid_count != 0 && !ShakeDebugEnabled()) {
    const thrust::host_vector<rbmd::Id> invalid_gids = d_first_invalid_gids;
    std::ostringstream oss;
    oss << "MPI SHAKE halo does not contain a complete water cluster: rank="
        << GET_RBMD_PARALLEL->_domdec->_current_rank << " epoch=" << epoch
        << " invalid_candidates=" << invalid_count << " first_cluster=("
        << invalid_gids[0] << "," << invalid_gids[1] << ","
        << invalid_gids[2] << ")";
    throw std::runtime_error(oss.str());
  }

  int global_invalid_count = 0;
  if (ShakeDebugEnabled()) {
    MPI_CHECK(MPI_Allreduce(&invalid_count, &global_invalid_count, 1, MPI_INT,
                            MPI_SUM, GET_RBMD_PARALLEL->_domdec->_mpi_comm));
  }
  if (global_invalid_count != 0) {
    thrust::host_vector<rbmd::Id> local_invalid_gids = d_first_invalid_gids;
    std::array<rbmd::Id, 3> local_invalid{
        local_invalid_gids[0], local_invalid_gids[1], local_invalid_gids[2]};
    std::vector<rbmd::Id> all_invalid(
        static_cast<std::size_t>(GET_RBMD_PARALLEL->_domdec->_total_ranks) * 3,
        rbmd::Id(-1));
    MPI_CHECK(MPI_Allgather(
        local_invalid.data(), 3, MPI_RBMD_ID, all_invalid.data(), 3,
        MPI_RBMD_ID, GET_RBMD_PARALLEL->_domdec->_mpi_comm));
    thrust::host_vector<rbmd::Id> h_ids(
        _device_data->_d_atoms_id.begin(),
        _device_data->_d_atoms_id.begin() + total_atoms);
    thrust::host_vector<rbmd::Real> h_px(
        _device_data->_d_px.begin(), _device_data->_d_px.begin() + total_atoms);
    thrust::host_vector<rbmd::Real> h_py(
        _device_data->_d_py.begin(), _device_data->_d_py.begin() + total_atoms);
    thrust::host_vector<rbmd::Real> h_pz(
        _device_data->_d_pz.begin(), _device_data->_d_pz.begin() + total_atoms);
    thrust::host_vector<int> h_num_angle(
        _device_data->d_num_angle.begin(),
        _device_data->d_num_angle.begin() + native_atoms);
    thrust::host_vector<rbmd::Id> h_angle_atom0(
        _device_data->d_angle_atom1.begin(),
        _device_data->d_angle_atom1.begin() + angle_slots);
    thrust::host_vector<rbmd::Id> h_angle_atom1(
        _device_data->d_angle_atom2.begin(),
        _device_data->d_angle_atom2.begin() + angle_slots);
    thrust::host_vector<rbmd::Id> h_angle_atom2(
        _device_data->d_angle_atom3.begin(),
        _device_data->d_angle_atom3.begin() + angle_slots);
    std::vector<int> local_native_counts(all_invalid.size(), 0);
    std::vector<int> local_replica_counts(all_invalid.size(), 0);
    std::vector<rbmd::Real> local_owner_x(all_invalid.size(), rbmd::Real(0));
    std::vector<rbmd::Real> local_owner_y(all_invalid.size(), rbmd::Real(0));
    std::vector<rbmd::Real> local_owner_z(all_invalid.size(), rbmd::Real(0));
    std::vector<int> local_owner_rank(all_invalid.size(), -1);
    for (std::size_t requested = 0; requested < all_invalid.size();
         ++requested) {
      const rbmd::Id gid = all_invalid[requested];
      if (gid < 0) {
        continue;
      }
      for (rbmd::Id idx = 0; idx < total_atoms; ++idx) {
        if (h_ids[static_cast<std::size_t>(idx)] != gid) {
          continue;
        }
        ++local_replica_counts[requested];
        if (idx < native_atoms) {
          ++local_native_counts[requested];
          local_owner_x[requested] = h_px[static_cast<std::size_t>(idx)];
          local_owner_y[requested] = h_py[static_cast<std::size_t>(idx)];
          local_owner_z[requested] = h_pz[static_cast<std::size_t>(idx)];
          local_owner_rank[requested] =
              GET_RBMD_PARALLEL->_domdec->_current_rank;
        }
      }
    }
    std::vector<int> global_native_counts(all_invalid.size(), 0);
    std::vector<int> global_replica_counts(all_invalid.size(), 0);
    std::vector<rbmd::Real> global_owner_x(all_invalid.size(), rbmd::Real(0));
    std::vector<rbmd::Real> global_owner_y(all_invalid.size(), rbmd::Real(0));
    std::vector<rbmd::Real> global_owner_z(all_invalid.size(), rbmd::Real(0));
    std::vector<int> global_owner_rank(all_invalid.size(), -1);
    MPI_CHECK(MPI_Allreduce(
        local_native_counts.data(), global_native_counts.data(),
        static_cast<int>(all_invalid.size()), MPI_INT, MPI_SUM,
        GET_RBMD_PARALLEL->_domdec->_mpi_comm));
    MPI_CHECK(MPI_Allreduce(
        local_replica_counts.data(), global_replica_counts.data(),
        static_cast<int>(all_invalid.size()), MPI_INT, MPI_SUM,
        GET_RBMD_PARALLEL->_domdec->_mpi_comm));
    MPI_CHECK(MPI_Allreduce(
        local_owner_x.data(), global_owner_x.data(),
        static_cast<int>(all_invalid.size()), MPI_RBMD_REAL, MPI_SUM,
        GET_RBMD_PARALLEL->_domdec->_mpi_comm));
    MPI_CHECK(MPI_Allreduce(
        local_owner_y.data(), global_owner_y.data(),
        static_cast<int>(all_invalid.size()), MPI_RBMD_REAL, MPI_SUM,
        GET_RBMD_PARALLEL->_domdec->_mpi_comm));
    MPI_CHECK(MPI_Allreduce(
        local_owner_z.data(), global_owner_z.data(),
        static_cast<int>(all_invalid.size()), MPI_RBMD_REAL, MPI_SUM,
        GET_RBMD_PARALLEL->_domdec->_mpi_comm));
    MPI_CHECK(MPI_Allreduce(
        local_owner_rank.data(), global_owner_rank.data(),
        static_cast<int>(all_invalid.size()), MPI_INT, MPI_MAX,
        GET_RBMD_PARALLEL->_domdec->_mpi_comm));
    if (invalid_count == 0) {
      throw std::runtime_error(
          "MPI SHAKE detected an incomplete cluster on a peer rank");
    }
    const thrust::host_vector<rbmd::Id>& invalid_gids = local_invalid_gids;
    std::array<int, 3> replica_counts{0, 0, 0};
    std::array<rbmd::Id, 3> first_replica{-1, -1, -1};
    for (std::size_t idx = 0; idx < h_ids.size(); ++idx) {
      const rbmd::Id gid = h_ids[idx];
      for (int member = 0; member < 3; ++member) {
        if (gid == invalid_gids[static_cast<std::size_t>(member)]) {
          ++replica_counts[static_cast<std::size_t>(member)];
          if (first_replica[static_cast<std::size_t>(member)] < 0) {
            first_replica[static_cast<std::size_t>(member)] =
                static_cast<rbmd::Id>(idx);
          }
        }
      }
    }
    rbmd::Id emitter_idx = -1;
    rbmd::Id emitter_gid = -1;
    for (rbmd::Id atom_idx = 0; atom_idx < native_atoms && emitter_idx < 0;
         ++atom_idx) {
      const int count = std::min(
          std::max(h_num_angle[static_cast<std::size_t>(atom_idx)], 0),
          angle_per_atom);
      for (int slot = 0; slot < count; ++slot) {
        const std::size_t flat =
            static_cast<std::size_t>(atom_idx) *
                static_cast<std::size_t>(angle_per_atom) +
            static_cast<std::size_t>(slot);
        if (h_angle_atom0[flat] == invalid_gids[0] &&
            h_angle_atom1[flat] == invalid_gids[1] &&
            h_angle_atom2[flat] == invalid_gids[2]) {
          emitter_idx = atom_idx;
          emitter_gid = h_ids[static_cast<std::size_t>(atom_idx)];
          break;
        }
      }
    }
    std::ostringstream oss;
    oss << "MPI SHAKE halo does not contain a complete water cluster: rank="
        << GET_RBMD_PARALLEL->_domdec->_current_rank << " epoch=" << epoch
        << " invalid_candidates=" << invalid_count << " first_cluster=("
        << invalid_gids[0] << "," << invalid_gids[1] << ","
        << invalid_gids[2] << ")"
        << " replicas=(" << replica_counts[0] << "," << replica_counts[1]
        << "," << replica_counts[2] << ")";
    const std::size_t rank_offset =
        static_cast<std::size_t>(
            GET_RBMD_PARALLEL->_domdec->_current_rank) *
        3;
    oss << " global_native=(" << global_native_counts[rank_offset] << ","
        << global_native_counts[rank_offset + 1] << ","
        << global_native_counts[rank_offset + 2] << ")"
        << " global_replicas=(" << global_replica_counts[rank_offset] << ","
        << global_replica_counts[rank_offset + 1] << ","
        << global_replica_counts[rank_offset + 2] << ")";
    for (int member = 0; member < 3; ++member) {
      const std::size_t offset =
          rank_offset + static_cast<std::size_t>(member);
      oss << std::setprecision(std::numeric_limits<rbmd::Real>::max_digits10)
          << " owner" << member << "_rank=" << global_owner_rank[offset]
          << "_pos=(" << global_owner_x[offset] << ","
          << global_owner_y[offset] << "," << global_owner_z[offset] << ")";
    }
    oss
        << " emitter_idx=" << emitter_idx << " emitter_gid=" << emitter_gid
        << " local_box=[(" << GET_RBMD_PARALLEL->_domdec->h_box->_coord_min[0]
        << "," << GET_RBMD_PARALLEL->_domdec->h_box->_coord_min[1] << ","
        << GET_RBMD_PARALLEL->_domdec->h_box->_coord_min[2] << "),("
        << GET_RBMD_PARALLEL->_domdec->h_box->_coord_max[0] << ","
        << GET_RBMD_PARALLEL->_domdec->h_box->_coord_max[1] << ","
        << GET_RBMD_PARALLEL->_domdec->h_box->_coord_max[2] << ")]";
    for (int member = 0; member < 3; ++member) {
      const rbmd::Id idx = first_replica[static_cast<std::size_t>(member)];
      if (idx >= 0) {
        const std::size_t pos = static_cast<std::size_t>(idx);
        oss << " member" << member << "_idx=" << idx << "_pos=("
            << h_px[pos] << "," << h_py[pos] << "," << h_pz[pos] << ")";
      }
    }
    throw std::runtime_error(oss.str());
  }
  if (cluster_count < 0 ||
      static_cast<std::size_t>(cluster_count) > angle_slots) {
    throw std::runtime_error("MPI SHAKE produced an invalid cluster count");
  }

  _device_data->_d_shake_cluster_idx0.resize(
      static_cast<std::size_t>(cluster_count));
  _device_data->_d_shake_cluster_idx1.resize(
      static_cast<std::size_t>(cluster_count));
  _device_data->_d_shake_cluster_idx2.resize(
      static_cast<std::size_t>(cluster_count));
  _shake_cluster_epoch = epoch;
  _shake_cluster_native_atoms = native_atoms;
  _shake_cluster_total_atoms = total_atoms;
  _shake_cluster_angle_per_atom = angle_per_atom;

#endif
}

void ShakeController::ZeroShakeCorrections() {
  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }
  const rbmd::Id total_atoms = linked_cell->_native_atoms_num;
  thrust::fill(_device_data->_d_shake_dx.begin(),
               _device_data->_d_shake_dx.begin() + total_atoms, rbmd::Real(0));
  thrust::fill(_device_data->_d_shake_dy.begin(),
               _device_data->_d_shake_dy.begin() + total_atoms, rbmd::Real(0));
  thrust::fill(_device_data->_d_shake_dz.begin(),
               _device_data->_d_shake_dz.begin() + total_atoms, rbmd::Real(0));
  thrust::fill(_device_data->_d_shake_dvx.begin(),
               _device_data->_d_shake_dvx.begin() + total_atoms, rbmd::Real(0));
  thrust::fill(_device_data->_d_shake_dvy.begin(),
               _device_data->_d_shake_dvy.begin() + total_atoms, rbmd::Real(0));
  thrust::fill(_device_data->_d_shake_dvz.begin(),
               _device_data->_d_shake_dvz.begin() + total_atoms, rbmd::Real(0));
}

void PrimeShakeGhostShadowStateAfterExchange() {
#ifdef USE_MPI
  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  if (total_atoms <= native_atoms) {
    return;
  }

  auto* device_data = DataManager::getInstance().getDeviceData().get();
  thrust::copy(device_data->_d_px.begin() + native_atoms,
               device_data->_d_px.begin() + total_atoms,
               device_data->_d_shake_px.begin() + native_atoms);
  thrust::copy(device_data->_d_py.begin() + native_atoms,
               device_data->_d_py.begin() + total_atoms,
               device_data->_d_shake_py.begin() + native_atoms);
  thrust::copy(device_data->_d_pz.begin() + native_atoms,
               device_data->_d_pz.begin() + total_atoms,
               device_data->_d_shake_pz.begin() + native_atoms);
  thrust::copy(device_data->_d_vx.begin() + native_atoms,
               device_data->_d_vx.begin() + total_atoms,
               device_data->_d_shake_vx.begin() + native_atoms);
  thrust::copy(device_data->_d_vy.begin() + native_atoms,
               device_data->_d_vy.begin() + total_atoms,
               device_data->_d_shake_vy.begin() + native_atoms);
  thrust::copy(device_data->_d_vz.begin() + native_atoms,
               device_data->_d_vz.begin() + total_atoms,
               device_data->_d_shake_vz.begin() + native_atoms);
#endif
}

void ValidateRefreshMpiShakeMembershipBeforeShakeA() {
#ifdef USE_MPI
  if (!ShakeDebugPreExchangeEnabled()) {
    return;
  }
  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }
  linked_cell->ClearDataHalo();
  auto parallel =
      RbmdParallelUntilLocator::GetInstance().GetRbmdParallelUntil();
  parallel->BalanceAndExchange();
  if (!parallel->PrepareForwardCoordinateExchange()) {
    throw std::runtime_error(
        "SHAKE debug pre-exchange could not prepare the forward halo plan");
  }
  PrimeShakeGhostShadowStateAfterExchange();
  if (ShakeDebugEnabled()) {
    std::ostringstream oss;
    oss << "[shake-debug] pre-shake BalanceAndExchange enabled";
    if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
      oss << " rank=" << GET_RBMD_PARALLEL->_domdec->_current_rank
          << "/" << GET_RBMD_PARALLEL->_domdec->_total_ranks;
    }
    EmitShakeDebugLine(oss.str());
  }
#endif
}

void ShakeController::RunLegacySerialShakeA() {
  if (ShakeDebugEnabled()) {
    std::ostringstream oss;
    oss << "[shake-debug] entering legacy ShakeA path";
#ifdef USE_MPI
    if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
      oss << " rank=" << GET_RBMD_PARALLEL->_domdec->_current_rank
          << "/" << GET_RBMD_PARALLEL->_domdec->_total_ranks;
    }
#endif
    EmitShakeDebugLine(oss.str());
  }

  const rbmd::Id num_atoms = *(_structure_info_data->_num_atoms);
  const Box shake_box = GetShakeBoxForMpi(*_box);
  auto atom_id_to_idx =
      LinkedCellLocator::GetInstance().GetLinkedCell()->_atom_id_to_idx;
  op::ShakeAOp<device::DEVICE_GPU>()(
      *(_structure_info_data->_num_angles), _dt, _fmt2v, shake_box,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_mass.data()),
      thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_id_vec.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_vx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_vy.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_vz.data()),
      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
      thrust::raw_pointer_cast(_device_data->_d_fz.data()),
      thrust::raw_pointer_cast(_device_data->_d_flagX.data()),
      thrust::raw_pointer_cast(_device_data->_d_flagY.data()),
      thrust::raw_pointer_cast(_device_data->_d_flagZ.data()));

  thrust::copy(_device_data->_d_shake_vx.begin(),
               _device_data->_d_shake_vx.begin() + num_atoms, _device_data->_d_vx.begin());
  thrust::copy(_device_data->_d_shake_vy.begin(),
               _device_data->_d_shake_vy.begin() + num_atoms, _device_data->_d_vy.begin());
  thrust::copy(_device_data->_d_shake_vz.begin(),
               _device_data->_d_shake_vz.begin() + num_atoms, _device_data->_d_vz.begin());

  thrust::copy(_device_data->_d_shake_px.begin(),
               _device_data->_d_shake_px.begin() + num_atoms, _device_data->_d_px.begin());
  thrust::copy(_device_data->_d_shake_py.begin(),
               _device_data->_d_shake_py.begin() + num_atoms, _device_data->_d_py.begin());
  thrust::copy(_device_data->_d_shake_pz.begin(),
               _device_data->_d_shake_pz.begin() + num_atoms, _device_data->_d_pz.begin());
}

void ShakeController::RunLegacySerialShakeB() {
  if (ShakeDebugEnabled()) {
    std::ostringstream oss;
    oss << "[shake-debug] entering legacy ShakeB path";
#ifdef USE_MPI
    if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
      oss << " rank=" << GET_RBMD_PARALLEL->_domdec->_current_rank
          << "/" << GET_RBMD_PARALLEL->_domdec->_total_ranks;
    }
#endif
    EmitShakeDebugLine(oss.str());
  }

  const rbmd::Id num_atoms = *(_structure_info_data->_num_atoms);
  const Box shake_box = GetShakeBoxForMpi(*_box);
  auto atom_id_to_idx =
      LinkedCellLocator::GetInstance().GetLinkedCell()->_atom_id_to_idx;
  op::ShakeBOp<device::DEVICE_GPU>()(
      *(_structure_info_data->_num_angles), _dt, _fmt2v, shake_box,
      thrust::raw_pointer_cast(atom_id_to_idx.data()),
      thrust::raw_pointer_cast(_device_data->_d_mass.data()),
      thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_angle_id_vec.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_vx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_vy.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_vz.data()),
      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
      thrust::raw_pointer_cast(_device_data->_d_fz.data()));

  thrust::copy(_device_data->_d_shake_vx.begin(),
               _device_data->_d_shake_vx.begin() + num_atoms, _device_data->_d_vx.begin());
  thrust::copy(_device_data->_d_shake_vy.begin(),
               _device_data->_d_shake_vy.begin() + num_atoms, _device_data->_d_vy.begin());
  thrust::copy(_device_data->_d_shake_vz.begin(),
               _device_data->_d_shake_vz.begin() + num_atoms, _device_data->_d_vz.begin());
}

void ShakeController::RunReplicatedShakeAKernel() {
  if (ShakeDebugEnabled()) {
    std::ostringstream oss;
    oss << "[shake-debug] entering replicated ShakeA path";
#ifdef USE_MPI
    if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
      oss << " rank=" << GET_RBMD_PARALLEL->_domdec->_current_rank
          << "/" << GET_RBMD_PARALLEL->_domdec->_total_ranks;
    }
#endif
    EmitShakeDebugLine(oss.str());
  }

  const Box shake_box = GetShakeBoxForMpi(*_box);
  const rbmd::Id num_cluster =
      static_cast<rbmd::Id>(_device_data->_d_shake_cluster_idx0.size());
  if (num_cluster <= 0) {
    return;
  }
  const rbmd::Id native_atoms =
      LinkedCellLocator::GetInstance().GetLinkedCell()->_native_atoms_num;
  op::ShakeAReplicatedOp<device::DEVICE_GPU>()(
      num_cluster, native_atoms, _dt, shake_box,
      thrust::raw_pointer_cast(_device_data->_d_mass.data()),
      thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx0.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx1.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx2.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_vx.data()),
      thrust::raw_pointer_cast(_device_data->_d_vy.data()),
      thrust::raw_pointer_cast(_device_data->_d_vz.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dy.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dz.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvy.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvz.data()));
}

void ShakeController::RunReplicatedShakeBKernel() {
  if (ShakeDebugEnabled()) {
    std::ostringstream oss;
    oss << "[shake-debug] entering replicated ShakeB path";
#ifdef USE_MPI
    if (GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec) {
      oss << " rank=" << GET_RBMD_PARALLEL->_domdec->_current_rank
          << "/" << GET_RBMD_PARALLEL->_domdec->_total_ranks;
    }
#endif
    EmitShakeDebugLine(oss.str());
  }

  const Box shake_box = GetShakeBoxForMpi(*_box);
  const rbmd::Id num_cluster =
      static_cast<rbmd::Id>(_device_data->_d_shake_cluster_idx0.size());
  if (num_cluster <= 0) {
    return;
  }
  const rbmd::Id native_atoms =
      LinkedCellLocator::GetInstance().GetLinkedCell()->_native_atoms_num;
  op::ShakeBReplicatedOp<device::DEVICE_GPU>()(
      num_cluster, native_atoms, shake_box,
      thrust::raw_pointer_cast(_device_data->_d_mass.data()),
      thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx0.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx1.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_cluster_idx2.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_vx.data()),
      thrust::raw_pointer_cast(_device_data->_d_vy.data()),
      thrust::raw_pointer_cast(_device_data->_d_vz.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvy.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvz.data()));
}

void ShakeController::SnapshotReplicatedShakeKernelIndices() {
  _device_data->_d_shake_last_kernel_idx0 = _device_data->_d_shake_cluster_idx0;
  _device_data->_d_shake_last_kernel_idx1 = _device_data->_d_shake_cluster_idx1;
  _device_data->_d_shake_last_kernel_idx2 = _device_data->_d_shake_cluster_idx2;
}

void ShakeController::ApplyReplicatedShakeCorrections(bool apply_position) {
  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell) {
    return;
  }
  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  if (native_atoms <= 0) {
    return;
  }
  const Box correction_box = GetShakeCorrectionBoxForMpi(*_box);
  op::ApplyShakeCorrectionsOp<device::DEVICE_GPU>()(
      native_atoms, correction_box, apply_position,
      thrust::raw_pointer_cast(_device_data->_d_shake_dx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dy.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dz.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvx.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvy.data()),
      thrust::raw_pointer_cast(_device_data->_d_shake_dvz.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_device_data->_d_vx.data()),
      thrust::raw_pointer_cast(_device_data->_d_vy.data()),
      thrust::raw_pointer_cast(_device_data->_d_vz.data()),
      thrust::raw_pointer_cast(_device_data->_d_flagX.data()),
      thrust::raw_pointer_cast(_device_data->_d_flagY.data()),
      thrust::raw_pointer_cast(_device_data->_d_flagZ.data()));

  if (apply_position) {
    thrust::copy(_device_data->_d_px.begin(), _device_data->_d_px.begin() + native_atoms,
                 _device_data->_d_shake_px.begin());
    thrust::copy(_device_data->_d_py.begin(), _device_data->_d_py.begin() + native_atoms,
                 _device_data->_d_shake_py.begin());
    thrust::copy(_device_data->_d_pz.begin(), _device_data->_d_pz.begin() + native_atoms,
                 _device_data->_d_shake_pz.begin());
  }
  thrust::copy(_device_data->_d_vx.begin(), _device_data->_d_vx.begin() + native_atoms,
               _device_data->_d_shake_vx.begin());
  thrust::copy(_device_data->_d_vy.begin(), _device_data->_d_vy.begin() + native_atoms,
               _device_data->_d_shake_vy.begin());
  thrust::copy(_device_data->_d_vz.begin(), _device_data->_d_vz.begin() + native_atoms,
               _device_data->_d_shake_vz.begin());
}

void ShakeController::EmitShakeResidualDebugCsv(const char* stage,
                                                ShakeCommStage resolve_stage) {
  if (!ShakeResidualCsvDebugEnabled() || stage == nullptr || !UseMpiOwnedShakePath()) {
    return;
  }

  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  if (!linked_cell || _device_data->_d_shake_cluster_idx0.empty()) {
    return;
  }

  (void)resolve_stage;

  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  if (total_atoms <= 0 || native_atoms <= 0) {
    return;
  }

  thrust::host_vector<rbmd::Id> h_idx0(_device_data->_d_shake_cluster_idx0.begin(),
                                       _device_data->_d_shake_cluster_idx0.end());
  thrust::host_vector<rbmd::Id> h_idx1(_device_data->_d_shake_cluster_idx1.begin(),
                                       _device_data->_d_shake_cluster_idx1.end());
  thrust::host_vector<rbmd::Id> h_idx2(_device_data->_d_shake_cluster_idx2.begin(),
                                       _device_data->_d_shake_cluster_idx2.end());
  thrust::host_vector<rbmd::Id> h_kernel_idx0(
      _device_data->_d_shake_last_kernel_idx0.begin(),
      _device_data->_d_shake_last_kernel_idx0.end());
  thrust::host_vector<rbmd::Id> h_kernel_idx1(
      _device_data->_d_shake_last_kernel_idx1.begin(),
      _device_data->_d_shake_last_kernel_idx1.end());
  thrust::host_vector<rbmd::Id> h_kernel_idx2(
      _device_data->_d_shake_last_kernel_idx2.begin(),
      _device_data->_d_shake_last_kernel_idx2.end());
  thrust::host_vector<rbmd::Id> h_ids(_device_data->_d_atoms_id.begin(),
                                      _device_data->_d_atoms_id.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_px(_device_data->_d_px.begin(),
                                       _device_data->_d_px.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_py(_device_data->_d_py.begin(),
                                       _device_data->_d_py.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_pz(_device_data->_d_pz.begin(),
                                       _device_data->_d_pz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_vx(_device_data->_d_vx.begin(),
                                       _device_data->_d_vx.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_vy(_device_data->_d_vy.begin(),
                                       _device_data->_d_vy.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_vz(_device_data->_d_vz.begin(),
                                       _device_data->_d_vz.begin() + total_atoms);
  thrust::host_vector<rbmd::Real> h_shake_dx(
      _device_data->_d_shake_dx.begin(),
      _device_data->_d_shake_dx.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_shake_dy(
      _device_data->_d_shake_dy.begin(),
      _device_data->_d_shake_dy.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_shake_dz(
      _device_data->_d_shake_dz.begin(),
      _device_data->_d_shake_dz.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_shake_dvx(
      _device_data->_d_shake_dvx.begin(),
      _device_data->_d_shake_dvx.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_shake_dvy(
      _device_data->_d_shake_dvy.begin(),
      _device_data->_d_shake_dvy.begin() + native_atoms);
  thrust::host_vector<rbmd::Real> h_shake_dvz(
      _device_data->_d_shake_dvz.begin(),
      _device_data->_d_shake_dvz.begin() + native_atoms);

  const Box shake_box = GetShakeBoxForMpi(*_box);
  const rbmd::Real bond_oh = 1.0;
  const rbmd::Real bond_hh =
      std::sqrt(2.0 * bond_oh * bond_oh -
                2.0 * bond_oh * bond_oh *
                    std::cos((109.4700 / 180.0) * M_PI));

  rbmd::Real max_abs_bond_err = 0.0;
  rbmd::Real sum_sq_bond_err = 0.0;
  rbmd::Real max_abs_hh_err = 0.0;
  rbmd::Real sum_sq_hh_err = 0.0;
  rbmd::Real max_abs_velocity_resid = 0.0;
  rbmd::Real sum_sq_velocity_resid = 0.0;
  rbmd::Real max_abs_shake_dx = 0.0;
  rbmd::Real sum_shake_dx2 = 0.0;
  rbmd::Real max_abs_shake_dv = 0.0;
  rbmd::Real sum_shake_dv2 = 0.0;
  rbmd::Id velocity_resid_count = 0;
  rbmd::Real worst_geom_error = -1.0;
  rbmd::Id worst_geom_gid0 = -1;
  rbmd::Id worst_geom_gid1 = -1;
  rbmd::Id worst_geom_gid2 = -1;
  rbmd::Id worst_geom_idx0 = -1;
  rbmd::Id worst_geom_idx1 = -1;
  rbmd::Id worst_geom_idx2 = -1;
  rbmd::Id worst_kernel_idx0 = -1;
  rbmd::Id worst_kernel_idx1 = -1;
  rbmd::Id worst_kernel_idx2 = -1;
  std::string worst_geom_region0 = "missing";
  std::string worst_geom_region1 = "missing";
  std::string worst_geom_region2 = "missing";
  std::string worst_kernel_region0 = "missing";
  std::string worst_kernel_region1 = "missing";
  std::string worst_kernel_region2 = "missing";
  rbmd::Real worst_geom_r01 = 0.0;
  rbmd::Real worst_geom_r12 = 0.0;
  rbmd::Real worst_geom_r20 = 0.0;
  rbmd::Real worst_geom_raw_r01 = 0.0;
  rbmd::Real worst_geom_raw_r12 = 0.0;
  rbmd::Real worst_geom_raw_r20 = 0.0;

  auto accumulate_velocity_resid = [&](rbmd::Id lhs, rbmd::Id rhs) {
    const std::size_t i = static_cast<std::size_t>(lhs);
    const std::size_t j = static_cast<std::size_t>(rhs);
    const rbmd::Real dx =
        HostMinImageDelta(h_px[i] - h_px[j], shake_box._length[0], shake_box._pbc_x);
    const rbmd::Real dy =
        HostMinImageDelta(h_py[i] - h_py[j], shake_box._length[1], shake_box._pbc_y);
    const rbmd::Real dz =
        HostMinImageDelta(h_pz[i] - h_pz[j], shake_box._length[2], shake_box._pbc_z);
    const rbmd::Real rij2 = dx * dx + dy * dy + dz * dz;
    if (rij2 <= rbmd::Real(0)) {
      return;
    }
    const rbmd::Real dvx = h_vx[i] - h_vx[j];
    const rbmd::Real dvy = h_vy[i] - h_vy[j];
    const rbmd::Real dvz = h_vz[i] - h_vz[j];
    const rbmd::Real resid =
        std::abs((dvx * dx + dvy * dy + dvz * dz) / std::sqrt(rij2));
    max_abs_velocity_resid = std::max(max_abs_velocity_resid, resid);
    sum_sq_velocity_resid += resid * resid;
    ++velocity_resid_count;
  };

  for (std::size_t cluster = 0; cluster < h_idx0.size(); ++cluster) {
    const rbmd::Id id0 = h_idx0[cluster];
    const rbmd::Id id1 = h_idx1[cluster];
    const rbmd::Id id2 = h_idx2[cluster];
    if (id0 < 0 || id1 < 0 || id2 < 0) {
      continue;
    }

    const std::size_t i0 = static_cast<std::size_t>(id0);
    const std::size_t i1 = static_cast<std::size_t>(id1);
    const std::size_t i2 = static_cast<std::size_t>(id2);

    const rbmd::Real r10 = std::sqrt(HostDistanceSquared(
        h_px[i0], h_py[i0], h_pz[i0], h_px[i1], h_py[i1], h_pz[i1], shake_box));
    const rbmd::Real r12 = std::sqrt(HostDistanceSquared(
        h_px[i2], h_py[i2], h_pz[i2], h_px[i1], h_py[i1], h_pz[i1], shake_box));
    const rbmd::Real r20 = std::sqrt(HostDistanceSquared(
        h_px[i0], h_py[i0], h_pz[i0], h_px[i2], h_py[i2], h_pz[i2], shake_box));

    const rbmd::Real bond_err0 = std::abs(r10 - bond_oh);
    const rbmd::Real bond_err1 = std::abs(r12 - bond_oh);
    const rbmd::Real hh_err = std::abs(r20 - bond_hh);
    const rbmd::Real cluster_geom_error = std::max(std::max(bond_err0, bond_err1), hh_err);
    max_abs_bond_err = std::max(max_abs_bond_err, std::max(bond_err0, bond_err1));
    sum_sq_bond_err += bond_err0 * bond_err0 + bond_err1 * bond_err1;
    max_abs_hh_err = std::max(max_abs_hh_err, hh_err);
    sum_sq_hh_err += hh_err * hh_err;

    if (cluster_geom_error > worst_geom_error) {
      worst_geom_error = cluster_geom_error;
      worst_geom_gid0 = h_ids[i0];
      worst_geom_gid1 = h_ids[i1];
      worst_geom_gid2 = h_ids[i2];
      worst_geom_idx0 = id0;
      worst_geom_idx1 = id1;
      worst_geom_idx2 = id2;
      if (cluster < h_kernel_idx0.size()) {
        worst_kernel_idx0 = h_kernel_idx0[cluster];
      }
      if (cluster < h_kernel_idx1.size()) {
        worst_kernel_idx1 = h_kernel_idx1[cluster];
      }
      if (cluster < h_kernel_idx2.size()) {
        worst_kernel_idx2 = h_kernel_idx2[cluster];
      }
      worst_geom_region0 = ShakeRegionName(id0, native_atoms, total_atoms);
      worst_geom_region1 = ShakeRegionName(id1, native_atoms, total_atoms);
      worst_geom_region2 = ShakeRegionName(id2, native_atoms, total_atoms);
      worst_kernel_region0 =
          ShakeRegionName(worst_kernel_idx0, native_atoms, total_atoms);
      worst_kernel_region1 =
          ShakeRegionName(worst_kernel_idx1, native_atoms, total_atoms);
      worst_kernel_region2 =
          ShakeRegionName(worst_kernel_idx2, native_atoms, total_atoms);
      worst_geom_r01 = r10;
      worst_geom_r12 = r12;
      worst_geom_r20 = r20;
      worst_geom_raw_r01 = std::sqrt(RawDistanceSquared(h_px, h_py, h_pz, id0, id1));
      worst_geom_raw_r12 = std::sqrt(RawDistanceSquared(h_px, h_py, h_pz, id2, id1));
      worst_geom_raw_r20 = std::sqrt(RawDistanceSquared(h_px, h_py, h_pz, id0, id2));
    }

    accumulate_velocity_resid(id0, id1);
    accumulate_velocity_resid(id2, id1);
    accumulate_velocity_resid(id0, id2);

    const rbmd::Id native_ids[3] = {id0, id1, id2};
    for (rbmd::Id native_id : native_ids) {
      if (native_id < 0 || native_id >= native_atoms) {
        continue;
      }
      const std::size_t idx = static_cast<std::size_t>(native_id);
      const rbmd::Real dx = h_shake_dx[idx];
      const rbmd::Real dy = h_shake_dy[idx];
      const rbmd::Real dz = h_shake_dz[idx];
      const rbmd::Real dvx = h_shake_dvx[idx];
      const rbmd::Real dvy = h_shake_dvy[idx];
      const rbmd::Real dvz = h_shake_dvz[idx];
      const rbmd::Real dx2 = dx * dx + dy * dy + dz * dz;
      const rbmd::Real dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
      max_abs_shake_dx = std::max(max_abs_shake_dx, std::sqrt(dx2));
      sum_shake_dx2 += dx2;
      max_abs_shake_dv = std::max(max_abs_shake_dv, std::sqrt(dv2));
      sum_shake_dv2 += dv2;
    }
  }

  const rbmd::Real rms_bond_err =
      h_idx0.empty() ? 0.0 : std::sqrt(sum_sq_bond_err / (2.0 * h_idx0.size()));
  const rbmd::Real rms_hh_err =
      h_idx0.empty() ? 0.0 : std::sqrt(sum_sq_hh_err / h_idx0.size());
  const rbmd::Real rms_velocity_resid =
      velocity_resid_count == 0
          ? 0.0
          : std::sqrt(sum_sq_velocity_resid /
                      static_cast<rbmd::Real>(velocity_resid_count));

  EnsureShakeResidualDebugLogDirectory();
#ifdef USE_MPI
  const int rank = GET_RBMD_PARALLEL && GET_RBMD_PARALLEL->_domdec
                       ? GET_RBMD_PARALLEL->_domdec->_current_rank
                       : 0;
#else
  const int rank = 0;
#endif
  const int pid = CurrentShakeProcessId();
  std::ofstream csv("logs/debug/shake_residual_debug_rank" + std::to_string(rank) +
                        "_pid" + std::to_string(pid) + ".csv",
                    std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,stage,cluster_count,max_abs_bond_err,rms_bond_err,"
           "max_abs_hh_err,rms_hh_err,max_abs_velocity_resid,"
           "rms_velocity_resid,max_abs_shake_dx,sum_shake_dx2,"
           "max_abs_shake_dv,sum_shake_dv2,worst_geom_gid0,worst_geom_gid1,"
           "worst_geom_gid2,worst_geom_idx0,worst_geom_idx1,worst_geom_idx2,"
           "worst_kernel_idx0,worst_kernel_idx1,worst_kernel_idx2,"
           "worst_geom_region0,worst_geom_region1,worst_geom_region2,"
           "worst_kernel_region0,worst_kernel_region1,worst_kernel_region2,"
           "worst_geom_r01,worst_geom_r12,worst_geom_r20,worst_geom_raw_r01,"
           "worst_geom_raw_r12,worst_geom_raw_r20\n";
  }

  csv << test_current_step << "," << rank << "," << stage << "," << h_idx0.size()
      << "," << max_abs_bond_err << "," << rms_bond_err << "," << max_abs_hh_err
      << "," << rms_hh_err << "," << max_abs_velocity_resid << ","
      << rms_velocity_resid << "," << max_abs_shake_dx << "," << sum_shake_dx2
      << "," << max_abs_shake_dv << "," << sum_shake_dv2 << ","
      << worst_geom_gid0 << "," << worst_geom_gid1 << "," << worst_geom_gid2
      << "," << worst_geom_idx0 << "," << worst_geom_idx1 << ","
      << worst_geom_idx2 << "," << worst_kernel_idx0 << ","
      << worst_kernel_idx1 << "," << worst_kernel_idx2 << ","
      << worst_geom_region0 << "," << worst_geom_region1 << ","
      << worst_geom_region2 << "," << worst_kernel_region0 << ","
      << worst_kernel_region1 << "," << worst_kernel_region2 << ","
      << worst_geom_r01 << "," << worst_geom_r12 << "," << worst_geom_r20
      << "," << worst_geom_raw_r01 << "," << worst_geom_raw_r12 << ","
      << worst_geom_raw_r20 << "\n";
}

void ShakeController::ShakeA() {
  if (!UseMpiOwnedShakePath()) {
    RunLegacySerialShakeA();
    return;
  }

  ValidateRefreshMpiShakeMembershipBeforeShakeA();

  auto stage_start = ShakeTimingClock::now();
  EnsureReplicatedShakeClusters();
  RecordShakeTiming("SHAKE-A-Cache", stage_start);

  // Normal halo exchange does not populate ghost shake shadow state.
  // Prime it from the current ghost position/velocity before inferring
  // periodic images for the first predicted-position forward sync.
  stage_start = ShakeTimingClock::now();
  PrimeShakeGhostShadowStateAfterExchange();
  ForwardSyncReplicatedShakeState(ShakeCommStage::PredictedPositionForward);
  RecordShakeTiming("SHAKE-A-InHalo", stage_start);
  if (ShakeResidualCsvDebugEnabled()) {
    SnapshotReplicatedShakeKernelIndices();
  }
  stage_start = ShakeTimingClock::now();
  ZeroShakeCorrections();
  RunReplicatedShakeAKernel();
  RecordShakeTiming("SHAKE-A-Kernel", stage_start);
  stage_start = ShakeTimingClock::now();
  ApplyReplicatedShakeCorrections(true);
  RecordShakeTiming("SHAKE-A-Apply", stage_start);
  stage_start = ShakeTimingClock::now();
  ForwardSyncReplicatedShakeState(ShakeCommStage::CorrectedStateForward);
  RecordShakeTiming("SHAKE-A-OutHalo", stage_start);
  EmitShakeResidualDebugCsv("post_shake_a", ShakeCommStage::CorrectedStateForward);
}

void ShakeController::ShakeB() {
  if (!UseMpiOwnedShakePath()) {
    RunLegacySerialShakeB();
    return;
  }

  auto stage_start = ShakeTimingClock::now();
  EnsureReplicatedShakeClusters();
  RecordShakeTiming("SHAKE-B-Cache", stage_start);

  stage_start = ShakeTimingClock::now();
  ForwardSyncReplicatedShakeState(ShakeCommStage::VelocityForward);
  RecordShakeTiming("SHAKE-B-InHalo", stage_start);
  if (ShakeResidualCsvDebugEnabled()) {
    SnapshotReplicatedShakeKernelIndices();
  }
  stage_start = ShakeTimingClock::now();
  ZeroShakeCorrections();
  RunReplicatedShakeBKernel();
  RecordShakeTiming("SHAKE-B-Kernel", stage_start);
  stage_start = ShakeTimingClock::now();
  ApplyReplicatedShakeCorrections(false);
  RecordShakeTiming("SHAKE-B-Apply", stage_start);
  stage_start = ShakeTimingClock::now();
  ForwardSyncReplicatedShakeState(ShakeCommStage::VelocityCorrectedForward);
  RecordShakeTiming("SHAKE-B-OutHalo", stage_start);
  EmitShakeResidualDebugCsv("post_shake_b", ShakeCommStage::VelocityCorrectedForward);
}
