#include "indirect_neighbour_communication_scheme.h"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "data_manager.h"
#include "common/neighbor_skin.h"
#include "neighbor_acquirer.h"

extern rbmd::Id test_current_step;

namespace {

int CheckedIntCount(rbmd::Id value, const char* field) {
  if (value > static_cast<rbmd::Id>(std::numeric_limits<int>::max())) {
    std::ostringstream oss;
    oss << field << " exceeds int range required by compacted index buffers: "
        << value;
    throw std::overflow_error(oss.str());
  }
  return static_cast<int>(value);
}

bool EnvFlagEnabled(const char* name) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return false;
  }
  return std::string(v) != "0";
}

bool NearlyEqualCutoff(rbmd::Real lhs, rbmd::Real rhs) {
  const rbmd::Real scale = std::max<rbmd::Real>(
      rbmd::Real(1), std::max(std::abs(lhs), std::abs(rhs)));
  return std::abs(lhs - rhs) <= scale * rbmd::Real(1e-6);
}

template <typename T>
std::vector<T> CopyDeviceBufferColumn(const char* buffer, size_t offset,
                                      int count) {
  if (buffer == nullptr || count <= 0) {
    return {};
  }
  const auto* raw = reinterpret_cast<const T*>(buffer + offset);
  thrust::device_ptr<const T> begin(raw);
  thrust::host_vector<T> host(static_cast<size_t>(count));
  thrust::copy(begin, begin + count, host.begin());
  return std::vector<T>(host.begin(), host.end());
}

constexpr int kMaxForwardFields = 12;

int PartnerReceiveSide(CommunicationPartner& partner, unsigned short d) {
  if (d >= 3) {
    throw std::out_of_range("Communication dimension is out of range");
  }
  const int offset = partner.getOffset()[d];
  if (offset < 0) {
    return 0;
  }
  if (offset > 0) {
    return 1;
  }
  throw std::logic_error(
      "Communication partner has no offset in the active dimension");
}

int OppositeReceiveSide(int receive_side) { return 1 - receive_side; }

std::vector<CommunicationPartner> SqueezePartnersByDirection(
    std::vector<CommunicationPartner> partners, unsigned short d) {
  std::vector<CommunicationPartner> squeezed_partners;
  std::vector<bool> used(partners.size(), false);

  for (std::size_t i = 0; i < partners.size(); ++i) {
    if (used[i]) {
      continue;
    }

    const int rank = partners[i].getRank();
    const int receive_side = PartnerReceiveSide(partners[i], d);
    CommunicationPartner combined = partners[i];
    for (std::size_t j = i + 1; j < partners.size(); ++j) {
      if (used[j] || partners[j].getRank() != rank ||
          PartnerReceiveSide(partners[j], d) != receive_side) {
        continue;
      }
      combined.add(partners[j]);
      used[j] = true;
    }
    squeezed_partners.push_back(combined);
  }

  return squeezed_partners;
}

struct ForwardFieldLayout {
  const rbmd::Real* source[kMaxForwardFields]{};
  rbmd::Real* destination[kMaxForwardFields]{};
  int shift_dimension[kMaxForwardFields]{};
  int count{0};

  ForwardFieldLayout() {
    for (int i = 0; i < kMaxForwardFields; ++i) {
      shift_dimension[i] = -1;
    }
  }

  void Add(thrust::device_vector<rbmd::Real>& values,
           int coordinate_dimension = -1) {
    if (count >= kMaxForwardFields) {
      throw std::logic_error("Forward ghost field bundle exceeds capacity");
    }
    source[count] = thrust::raw_pointer_cast(values.data());
    destination[count] = thrust::raw_pointer_cast(values.data());
    shift_dimension[count] = coordinate_dimension;
    ++count;
  }
};

ForwardFieldLayout MakeForwardFieldLayout(
    ForwardGhostState state, const std::shared_ptr<DeviceData>& data) {
  if (!data) {
    throw std::runtime_error("Forward ghost state has no device data");
  }

  ForwardFieldLayout layout;
  const auto add_position = [&]() {
    layout.Add(data->_d_px, 0);
    layout.Add(data->_d_py, 1);
    layout.Add(data->_d_pz, 2);
  };
  const auto add_shadow_position = [&]() {
    layout.Add(data->_d_shake_px, 0);
    layout.Add(data->_d_shake_py, 1);
    layout.Add(data->_d_shake_pz, 2);
  };
  const auto add_velocity = [&]() {
    layout.Add(data->_d_vx);
    layout.Add(data->_d_vy);
    layout.Add(data->_d_vz);
  };
  const auto add_shadow_velocity = [&]() {
    layout.Add(data->_d_shake_vx);
    layout.Add(data->_d_shake_vy);
    layout.Add(data->_d_shake_vz);
  };

  switch (state) {
    case ForwardGhostState::Coordinates:
      add_position();
      break;
    case ForwardGhostState::PredictedPositionForward:
      add_position();
      add_shadow_position();
      add_velocity();
      break;
    case ForwardGhostState::CorrectedStateForward:
      add_position();
      add_shadow_position();
      add_velocity();
      add_shadow_velocity();
      break;
    case ForwardGhostState::VelocityForward:
      add_velocity();
      break;
    case ForwardGhostState::VelocityCorrectedForward:
      add_velocity();
      add_shadow_velocity();
      break;
    default:
      throw std::invalid_argument("Unknown forward ghost state bundle");
  }
  return layout;
}

struct PackForwardFields {
  const rbmd::Id* indices;
  const rbmd::Real* shift_x;
  const rbmd::Real* shift_y;
  const rbmd::Real* shift_z;
  const rbmd::Real* source[kMaxForwardFields];
  int shift_dimension[kMaxForwardFields];
  rbmd::Real* packed_values;
  rbmd::Id count;
  int field_count;

  __host__ __device__ void operator()(rbmd::Id i) const {
    const rbmd::Id src = indices[i];
    for (int field = 0; field < field_count; ++field) {
      rbmd::Real value = source[field][src];
      switch (shift_dimension[field]) {
        case 0:
          value += shift_x[i];
          break;
        case 1:
          value += shift_y[i];
          break;
        case 2:
          value += shift_z[i];
          break;
        default:
          break;
      }
      packed_values[static_cast<rbmd::Id>(field) * count + i] = value;
    }
  }
};

struct ApplyForwardFields {
  const rbmd::Real* packed_values;
  const rbmd::Id* indices;
  rbmd::Real* destination[kMaxForwardFields];
  rbmd::Id count;
  int field_count;

  __host__ __device__ void operator()(rbmd::Id i) const {
    const rbmd::Id dst = indices[i];
    if (dst < 0) {
      return;
    }
    for (int field = 0; field < field_count; ++field) {
      destination[field][dst] =
          packed_values[static_cast<rbmd::Id>(field) * count + i];
    }
  }
};

int EnvIntOrDefault(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return fallback;
  }
  const int parsed = std::atoi(v);
  return parsed > 0 ? parsed : fallback;
}

int CurrentProcessId() {
#if defined(_WIN32)
  return static_cast<int>(_getpid());
#else
  return static_cast<int>(::getpid());
#endif
}

void EnsureExchangeRouteLogDirectory() {
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

struct ExchangeRouteTraceConfig {
  bool route_enabled{false};
  bool leaving_consistency_enabled{false};
  bool log_zero_counts{false};
  int every{1};
  int leaving_consistency_abort_code{88};
  std::vector<rbmd::Id> gids{};
};

struct ExchangeStageStatsConfig {
  bool enabled{false};
  int every{1};
};

std::vector<rbmd::Id> ParseGidListFromEnv(const char* name) {
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

ExchangeRouteTraceConfig ParseExchangeRouteTraceConfig() {
  ExchangeRouteTraceConfig cfg;
  cfg.route_enabled = EnvFlagEnabled("RBMD_DEBUG_EXCHANGE_ROUTE");
  cfg.leaving_consistency_enabled =
      EnvFlagEnabled("RBMD_DEBUG_EXCHANGE_ASSERT_LEAVING");
  cfg.every = EnvIntOrDefault("RBMD_DEBUG_EXCHANGE_ROUTE_EVERY", 1);
  cfg.log_zero_counts = EnvFlagEnabled("RBMD_DEBUG_EXCHANGE_ROUTE_LOG_ZERO");
  cfg.leaving_consistency_abort_code =
      EnvIntOrDefault("RBMD_DEBUG_EXCHANGE_ASSERT_ABORT_CODE", 88);

  cfg.gids = ParseGidListFromEnv("RBMD_DEBUG_EXCHANGE_ROUTE_GIDS");
  if (cfg.gids.empty()) {
    cfg.gids = ParseGidListFromEnv("RBMD_DEBUG_TRACK_GIDS");
  }
  if (cfg.gids.empty()) {
    cfg.route_enabled = false;
  }
  return cfg;
}

ExchangeStageStatsConfig ParseExchangeStageStatsConfig() {
  ExchangeStageStatsConfig cfg;
  cfg.enabled = EnvFlagEnabled("RBMD_DEBUG_EXCHANGE_STAGE_STATS");
  cfg.every = EnvIntOrDefault("RBMD_DEBUG_EXCHANGE_STAGE_STATS_EVERY", 1);
  return cfg;
}

const ExchangeRouteTraceConfig& GetExchangeRouteTraceConfig() {
  static const ExchangeRouteTraceConfig cfg = ParseExchangeRouteTraceConfig();
  return cfg;
}

const ExchangeStageStatsConfig& GetExchangeStageStatsConfig() {
  static const ExchangeStageStatsConfig cfg = ParseExchangeStageStatsConfig();
  return cfg;
}

struct IsKeptMark {
  __host__ __device__ bool operator()(int mark) const { return mark == 0; }
};

struct RowMajorGatherIndex {
  const int* keep;
  int width;

  __host__ __device__ size_t operator()(size_t e) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int new_i = static_cast<int>(e / static_cast<size_t>(width));
    const int k = static_cast<int>(
        e - static_cast<size_t>(new_i) * static_cast<size_t>(width));
    const int old_i = keep[new_i];
    return static_cast<size_t>(old_i) * static_cast<size_t>(width) +
           static_cast<size_t>(k);
#else
    // host 路径不应执行（这些 thrust 算法针对 device_vector 会走 device 后端）。
    // 这里返回 0 仅用于满足编译期可调用性要求。
    (void)e;
    return 0;
#endif
  }
};

bool ShouldTraceExchangeRoute(rbmd::Id step) {
  const auto& cfg = GetExchangeRouteTraceConfig();
  if (!cfg.route_enabled || cfg.every <= 0) {
    return false;
  }
  return (step % cfg.every) == 0;
}

bool ShouldCheckLeavingConsistency(rbmd::Id step) {
  const auto& cfg = GetExchangeRouteTraceConfig();
  if (!cfg.leaving_consistency_enabled || cfg.every <= 0) {
    return false;
  }
  return (step % cfg.every) == 0;
}

bool ShouldTraceExchangeStageStats(rbmd::Id step) {
  const auto& cfg = GetExchangeStageStatsConfig();
  if (!cfg.enabled || cfg.every <= 0) {
    return false;
  }
  return (step % cfg.every) == 0;
}

const char* MsgTypeName(MessageType msg_type) {
  switch (msg_type) {
    case LEAVING_ONLY:
      return "leaving";
    case HALO_COPIES:
      return "halo";
    case LEAVING_AND_HALO_COPIES:
      return "leaving+halo";
    case FORCES:
      return "forces";
    default:
      return "unknown";
  }
}

const char* PhaseName(bool leaving_phase) {
  return leaving_phase ? "leaving" : "halo";
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

void AppendExchangeStageStatsRow(const char* stage, rbmd::Id step, uint32_t epoch,
                                 int rank, rbmd::Id native_atoms,
                                 rbmd::Id ghost_atoms, rbmd::Id total_atoms,
                                 rbmd::Id send_atoms, rbmd::Id recv_atoms,
                                 long long send_bytes,
                                 long long recv_bytes) {
  EnsureExchangeRouteLogDirectory();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/exchange_stage_stats_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,epoch,stage,rank,native_atoms,ghost_atoms,total_atoms,"
           "send_atoms,recv_atoms,send_bytes,recv_bytes\n";
  }
  csv << step << "," << epoch << "," << stage << "," << rank << ","
      << native_atoms << "," << ghost_atoms << "," << total_atoms << ","
      << send_atoms << "," << recv_atoms << "," << send_bytes << ","
      << recv_bytes << "\n";
}

void AppendTrackedGidPresenceRows(const ExchangeRouteTraceConfig& cfg,
                                  rbmd::Id step,
                                  uint32_t epoch,
                                  const char* checkpoint,
                                  int rank,
                                  rbmd::Id nlocal,
                                  rbmd::Id nghost,
                                  rbmd::Id ntotal,
                                  const thrust::device_vector<rbmd::Id>& d_atoms_id,
                                  MPI_Comm comm) {
  if (cfg.gids.empty()) {
    return;
  }
  if (!cfg.route_enabled && !cfg.leaving_consistency_enabled) {
    return;
  }

  std::vector<int> local_counts(cfg.gids.size(), 0);
  std::vector<int> global_counts(cfg.gids.size(), 0);
  std::vector<rbmd::Id> first_indices(cfg.gids.size(), rbmd::Id(-1));

  std::unordered_map<rbmd::Id, size_t> gid_to_slot;
  gid_to_slot.reserve(cfg.gids.size());
  for (size_t i = 0; i < cfg.gids.size(); ++i) {
    gid_to_slot.emplace(cfg.gids[i], i);
  }

  const rbmd::Id clamped_total = std::max<rbmd::Id>(0, ntotal);
  const size_t scan_size =
      std::min(static_cast<size_t>(clamped_total), d_atoms_id.size());
  if (scan_size > 0) {
    thrust::host_vector<rbmd::Id> h_ids(d_atoms_id.begin(),
                                        d_atoms_id.begin() + scan_size);
    for (size_t idx = 0; idx < h_ids.size(); ++idx) {
      const auto it = gid_to_slot.find(h_ids[idx]);
      if (it == gid_to_slot.end()) {
        continue;
      }
      const size_t slot = it->second;
      local_counts[slot] += 1;
      if (first_indices[slot] < 0) {
        first_indices[slot] = static_cast<rbmd::Id>(idx);
      }
    }
  }

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  for (size_t i = 0; i < local_counts.size(); ++i) {
    int global_count = local_counts[i];
    if (mpi_initialized) {
      MPI_Allreduce(&local_counts[i], &global_count, 1, MPI_INT, MPI_SUM, comm);
    }
    global_counts[i] = global_count;
  }

  EnsureExchangeRouteLogDirectory();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/exchange_gid_presence_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,epoch,checkpoint,rank,nlocal,nghost,ntotal,gid,local_count,"
           "global_count,first_idx,first_region\n";
  }

  for (size_t i = 0; i < cfg.gids.size(); ++i) {
    if (!cfg.log_zero_counts && local_counts[i] == 0 && global_counts[i] == 0) {
      continue;
    }
    const rbmd::Id first_idx = first_indices[i];
    csv << step << "," << epoch << "," << checkpoint << "," << rank << ","
        << nlocal << "," << nghost << "," << ntotal << "," << cfg.gids[i]
        << "," << local_counts[i] << "," << global_counts[i] << ","
        << first_idx << ","
        << AtomRegionName(first_idx, nlocal, ntotal) << "\n";
  }
}

std::vector<int> ExtractTrackedCountsFromLeavingBuffer(
    const char* device_ptr, int bytes, int atom_count,
    const std::vector<rbmd::Id>& tracked_gids) {
  std::vector<int> counts(tracked_gids.size(), 0);
  if (!device_ptr || bytes <= 0 || atom_count <= 0 || tracked_gids.empty()) {
    return counts;
  }

  const auto atom_style =
      DataManager::getInstance().getConfigData()->Get<std::string>(
          "atom_style", "init_configuration", "read_data");
  if (atom_style != "full") {
    const size_t id_bytes =
        static_cast<size_t>(atom_count) * sizeof(rbmd::Id);
    if (static_cast<size_t>(bytes) < id_bytes) {
      return counts;
    }

    std::vector<rbmd::Id> ids(static_cast<size_t>(atom_count), 0);
    CHECK_RUNTIME(MEMCPY(ids.data(), device_ptr, id_bytes, D2H));

    std::unordered_map<rbmd::Id, size_t> gid_to_index;
    gid_to_index.reserve(tracked_gids.size());
    for (size_t i = 0; i < tracked_gids.size(); ++i) {
      gid_to_index.emplace(tracked_gids[i], i);
    }

    for (const rbmd::Id gid : ids) {
      const auto it = gid_to_index.find(gid);
      if (it != gid_to_index.end()) {
        counts[it->second] += 1;
      }
    }
    return counts;
  }

  if (bytes % static_cast<int>(sizeof(double)) != 0) {
    return counts;
  }

  std::vector<double> host_buf(static_cast<size_t>(bytes) / sizeof(double), 0.0);
  CHECK_RUNTIME(MEMCPY(host_buf.data(), device_ptr, static_cast<size_t>(bytes), D2H));
  if (host_buf.empty()) {
    return counts;
  }

  int size_exchange = static_cast<int>(host_buf[0]);
  const int fallback = atom_count > 0
                           ? static_cast<int>(host_buf.size() /
                                              static_cast<size_t>(atom_count))
                           : 0;
  if (size_exchange <= 0 ||
      static_cast<size_t>(size_exchange) * static_cast<size_t>(atom_count) >
          host_buf.size()) {
    size_exchange = fallback;
  }
  if (size_exchange <= 7) {
    return counts;
  }

  std::unordered_map<rbmd::Id, size_t> gid_to_index;
  gid_to_index.reserve(tracked_gids.size());
  for (size_t i = 0; i < tracked_gids.size(); ++i) {
    gid_to_index.emplace(tracked_gids[i], i);
  }

  union UBuf {
    double d;
    int64_t i;
  };

  for (int atom = 0; atom < atom_count; ++atom) {
    const size_t gid_offset =
        static_cast<size_t>(atom) * static_cast<size_t>(size_exchange) + 7U;
    if (gid_offset >= host_buf.size()) {
      break;
    }
    UBuf u{};
    u.d = host_buf[gid_offset];
    const rbmd::Id gid = static_cast<rbmd::Id>(u.i);
    const auto it = gid_to_index.find(gid);
    if (it != gid_to_index.end()) {
      counts[it->second] += 1;
    }
  }
  return counts;
}

std::vector<int> ExtractTrackedCountsFromHaloBuffer(
    const char* device_ptr, int bytes, int atom_count,
    const std::vector<rbmd::Id>& tracked_gids) {
  std::vector<int> counts(tracked_gids.size(), 0);
  if (!device_ptr || bytes <= 0 || atom_count <= 0 || tracked_gids.empty()) {
    return counts;
  }

  const size_t id_bytes =
      static_cast<size_t>(atom_count) * sizeof(rbmd::Id);
  if (static_cast<size_t>(bytes) < id_bytes) {
    return counts;
  }

  std::vector<rbmd::Id> ids(static_cast<size_t>(atom_count), 0);
  CHECK_RUNTIME(MEMCPY(ids.data(), device_ptr, id_bytes, D2H));

  std::unordered_map<rbmd::Id, size_t> gid_to_index;
  gid_to_index.reserve(tracked_gids.size());
  for (size_t i = 0; i < tracked_gids.size(); ++i) {
    gid_to_index.emplace(tracked_gids[i], i);
  }

  for (const rbmd::Id gid : ids) {
    const auto it = gid_to_index.find(gid);
    if (it != gid_to_index.end()) {
      counts[it->second] += 1;
    }
  }
  return counts;
}

bool MpiIsInitializedSafe() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0;
}

std::vector<int> CountTrackedMarkedLeavingAtoms(
    const std::vector<rbmd::Id>& tracked_gids,
    const thrust::device_vector<int>& staged_marks,
    const thrust::device_vector<rbmd::Id>& atom_ids,
    rbmd::Id native_atoms) {
  std::vector<int> counts(tracked_gids.size(), 0);
  if (tracked_gids.empty() || native_atoms <= 0 || staged_marks.empty() ||
      atom_ids.empty()) {
    return counts;
  }

  const size_t n = std::min(
      static_cast<size_t>(native_atoms),
      std::min(staged_marks.size(), atom_ids.size()));
  if (n == 0) {
    return counts;
  }

  std::vector<int> h_marks(n, 0);
  std::vector<rbmd::Id> h_ids(n, 0);
  CHECK_RUNTIME(
      MEMCPY(h_marks.data(), thrust::raw_pointer_cast(staged_marks.data()),
             n * sizeof(int), D2H));
  CHECK_RUNTIME(
      MEMCPY(h_ids.data(), thrust::raw_pointer_cast(atom_ids.data()),
             n * sizeof(rbmd::Id), D2H));

  std::unordered_map<rbmd::Id, size_t> gid_to_index;
  gid_to_index.reserve(tracked_gids.size());
  for (size_t i = 0; i < tracked_gids.size(); ++i) {
    gid_to_index.emplace(tracked_gids[i], i);
  }

  for (size_t i = 0; i < n; ++i) {
    if (h_marks[i] == 0) {
      continue;
    }
    const auto it = gid_to_index.find(h_ids[i]);
    if (it != gid_to_index.end()) {
      counts[it->second] += 1;
    }
  }
  return counts;
}

std::vector<rbmd::Id> CollectTrackedFirstNativeIndices(
    const std::vector<rbmd::Id>& tracked_gids,
    const thrust::device_vector<rbmd::Id>& atom_ids, rbmd::Id native_atoms) {
  std::vector<rbmd::Id> first_indices(tracked_gids.size(), rbmd::Id(-1));
  if (tracked_gids.empty() || native_atoms <= 0 || atom_ids.empty()) {
    return first_indices;
  }

  const size_t n =
      std::min(static_cast<size_t>(native_atoms), atom_ids.size());
  if (n == 0) {
    return first_indices;
  }

  std::vector<rbmd::Id> h_ids(n, 0);
  CHECK_RUNTIME(
      MEMCPY(h_ids.data(), thrust::raw_pointer_cast(atom_ids.data()),
             n * sizeof(rbmd::Id), D2H));

  std::unordered_map<rbmd::Id, size_t> gid_to_index;
  gid_to_index.reserve(tracked_gids.size());
  for (size_t i = 0; i < tracked_gids.size(); ++i) {
    gid_to_index.emplace(tracked_gids[i], i);
  }

  for (size_t i = 0; i < n; ++i) {
    const rbmd::Id gid = h_ids[i];
    const auto it = gid_to_index.find(gid);
    if (it == gid_to_index.end()) {
      continue;
    }
    rbmd::Id& old_idx = first_indices[it->second];
    if (old_idx < 0) {
      old_idx = static_cast<rbmd::Id>(i);
    }
  }
  return first_indices;
}

bool HasLeavingConsistencyMismatch(const std::vector<int>& marked_counts,
                                   const std::vector<int>& sent_counts) {
  const size_t n = std::min(marked_counts.size(), sent_counts.size());
  for (size_t i = 0; i < n; ++i) {
    if (marked_counts[i] != sent_counts[i]) {
      return true;
    }
  }
  return false;
}

std::string FormatTrackedSendPeers(
    const std::vector<std::pair<int, int>>& peer_counts) {
  if (peer_counts.empty()) {
    return "-";
  }
  std::vector<std::pair<int, int>> ordered = peer_counts;
  std::sort(ordered.begin(), ordered.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
              return a.first < b.first;
            });
  std::ostringstream oss;
  for (size_t i = 0; i < ordered.size(); ++i) {
    if (i > 0) {
      oss << "|";
    }
    oss << ordered[i].first << ":" << ordered[i].second;
  }
  return oss.str();
}

void DumpLeavingConsistencyContextAndExit(
    const ExchangeRouteTraceConfig& cfg, rbmd::Id step, uint32_t epoch,
    const std::string& check_point, int current_rank, rbmd::Id native_atoms,
    rbmd::Id ghost_atoms, rbmd::Id total_atoms, rbmd::Id marked_total,
    rbmd::Id sent_total,
    const std::vector<int>& marked_counts, const std::vector<int>& sent_counts,
    const std::vector<rbmd::Id>& old_indices,
    const std::vector<std::vector<std::pair<int, int>>>& sent_peers) {
  EnsureExchangeRouteLogDirectory();
  const int pid = CurrentProcessId();
  const std::string filename =
      "logs/debug/leaving_send_consistency_abort_rank" +
      std::to_string(current_rank) + "_pid" + std::to_string(pid) + "_step" +
      std::to_string(step) + ".log";

  std::ofstream log(filename, std::ios::out | std::ios::trunc);
  if (log.is_open()) {
    log << "=== Leaving Send Consistency Abort Context ===\n";
    log << "step: " << step << "\n";
    log << "epoch: " << epoch << "\n";
    log << "check_point: " << check_point << "\n";
    log << "rank: " << current_rank << "\n";
    log << "pid: " << pid << "\n";
    log << "native_atoms: " << native_atoms << "\n";
    log << "ghost_atoms: " << ghost_atoms << "\n";
    log << "total_atoms: " << total_atoms << "\n";
    log << "marked_total: " << marked_total << "\n";
    log << "sent_total: " << sent_total << "\n";
    log << "aggregate_mismatch: " << ((marked_total != sent_total) ? 1 : 0)
        << "\n";
    log << "abort_code: " << cfg.leaving_consistency_abort_code << "\n\n";

    if (!cfg.gids.empty()) {
      log << "gid,old_idx,marked_count,sent_count,in_sendlist,send_peers,mismatch\n";
      const size_t n =
          std::min(cfg.gids.size(), std::min(marked_counts.size(), sent_counts.size()));
      for (size_t i = 0; i < n; ++i) {
        const rbmd::Id old_idx =
            i < old_indices.size() ? old_indices[i] : rbmd::Id(-1);
        const int mismatch = (marked_counts[i] != sent_counts[i]) ? 1 : 0;
        const int in_sendlist = sent_counts[i] > 0 ? 1 : 0;
        const std::string send_peer_desc =
            i < sent_peers.size() ? FormatTrackedSendPeers(sent_peers[i]) : "-";
        log << cfg.gids[i] << "," << old_idx << "," << marked_counts[i] << ","
            << sent_counts[i] << "," << in_sendlist << "," << send_peer_desc
            << "," << mismatch << "\n";
      }
    } else {
      log << "tracked_gid_table: disabled (no tracked gids configured)\n";
    }
    log.close();
  }

  std::cerr << "[ABORT] leaving-send consistency mismatch, context dumped: "
            << filename << std::endl;
  if (MpiIsInitializedSafe()) {
    MPI_Abort(MPI_COMM_WORLD, cfg.leaving_consistency_abort_code);
  }
  std::exit(cfg.leaving_consistency_abort_code);
}

void AppendExchangeRouteRows(const ExchangeRouteTraceConfig& cfg, rbmd::Id step,
                             uint32_t epoch, unsigned short dim,
                             bool leaving_phase, MessageType msg_type,
                             const char* direction, int rank, int peer_rank,
                             int atom_count, int byte_count,
                             const std::vector<int>& gid_counts) {
  if (!cfg.route_enabled || cfg.gids.empty()) {
    return;
  }

  EnsureExchangeRouteLogDirectory();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/exchange_gid_route_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,epoch,dim,phase,msg_type,direction,rank,peer_rank,gid,count,"
           "atom_count,byte_count\n";
  }

  for (size_t i = 0; i < cfg.gids.size() && i < gid_counts.size(); ++i) {
    if (!cfg.log_zero_counts && gid_counts[i] == 0) {
      continue;
    }
    csv << step << "," << epoch << "," << dim << ","
        << PhaseName(leaving_phase) << "," << MsgTypeName(msg_type) << ","
        << direction << "," << rank << "," << peer_rank << ","
        << cfg.gids[i] << "," << gid_counts[i] << ","
        << atom_count << "," << byte_count << "\n";
  }
}

}  // namespace

MessageType IndirectNeighbourCommunicationScheme::PhaseToMessageType(
    Phase phase) {
  switch (phase) {
    case Phase::Leaving:
      return LEAVING_ONLY;
    case Phase::Halo:
      return HALO_COPIES;
    default:
      return LEAVING_ONLY;
  }
}

void IndirectNeighbourCommunicationScheme::InvalidateGhostCoordinateExchange() {
  _forward_stages.clear();
  _forward_coordinates_valid = false;
}

void IndirectNeighbourCommunicationScheme::CaptureGhostPhase(
    unsigned short d, const PhaseExchangeResult& result,
    LinkedCell* moleculeContainer, DomainDecomposition* domainDecomp) {
  if (!_capture_forward_coordinates || !_pending_forward_valid ||
      moleculeContainer == nullptr || domainDecomp == nullptr ||
      d >= _pending_forward_stages.size()) {
    return;
  }

  auto& neighbours = (*_neighbours)[d];
  auto& stage = _pending_forward_stages[d];
  stage.clear();
  stage.resize(neighbours.size());

  const rbmd::Id total_atoms = moleculeContainer->_total_atoms_num;
  if (!_device_data || total_atoms < 0 ||
      _device_data->_d_atoms_id.size() < static_cast<size_t>(total_atoms)) {
    _pending_forward_valid = false;
    return;
  }

  thrust::host_vector<rbmd::Id> ids(static_cast<size_t>(total_atoms));
  thrust::host_vector<rbmd::Real> px(static_cast<size_t>(total_atoms));
  thrust::host_vector<rbmd::Real> py(static_cast<size_t>(total_atoms));
  thrust::host_vector<rbmd::Real> pz(static_cast<size_t>(total_atoms));
  thrust::copy_n(_device_data->_d_atoms_id.begin(), total_atoms, ids.begin());
  thrust::copy_n(_device_data->_d_px.begin(), total_atoms, px.begin());
  thrust::copy_n(_device_data->_d_py.begin(), total_atoms, py.begin());
  thrust::copy_n(_device_data->_d_pz.begin(), total_atoms, pz.begin());

  std::unordered_map<rbmd::Id, std::vector<size_t>> local_indices;
  local_indices.reserve(static_cast<size_t>(total_atoms));
  for (size_t i = 0; i < static_cast<size_t>(total_atoms); ++i) {
    local_indices[ids[i]].push_back(i);
  }

  for (size_t i = 0; i < neighbours.size(); ++i) {
    auto& pending = stage[i];
    auto& partner = neighbours[i];
    pending.rank = partner.getRank();
    pending.receive_side = PartnerReceiveSide(partner, d);
    if (pending.rank == domainDecomp->_current_rank) {
      continue;
    }

    const int send_count = i < result.send_counts.size()
                               ? result.send_counts[i]
                               : 0;
    const int recv_count = i < result.recv_counts.size()
                               ? result.recv_counts[i]
                               : 0;
    const size_t send_min_bytes = static_cast<size_t>(send_count) *
        (sizeof(rbmd::Id) * 2 + sizeof(rbmd::Real) * 3);
    const size_t recv_min_bytes = static_cast<size_t>(recv_count) *
        (sizeof(rbmd::Id) * 2 + sizeof(rbmd::Real) * 3);
    if ((send_count > 0 &&
         (i >= result.send_bytes.size() ||
          static_cast<size_t>(result.send_bytes[i]) < send_min_bytes)) ||
        (recv_count > 0 &&
         (i >= result.recv_bytes.size() ||
          static_cast<size_t>(result.recv_bytes[i]) < recv_min_bytes))) {
      _pending_forward_valid = false;
      return;
    }

    const char* send_buffer = partner.GetSendBufferPtr(HALO_COPIES);
    pending.send_ids =
        CopyDeviceBufferColumn<rbmd::Id>(send_buffer, 0, send_count);
    const size_t send_position_offset =
        static_cast<size_t>(send_count) * sizeof(rbmd::Id) * 2;
    const auto send_px = CopyDeviceBufferColumn<rbmd::Real>(
        send_buffer, send_position_offset, send_count);
    const auto send_py = CopyDeviceBufferColumn<rbmd::Real>(
        send_buffer,
        send_position_offset + static_cast<size_t>(send_count) *
                                   sizeof(rbmd::Real),
        send_count);
    const auto send_pz = CopyDeviceBufferColumn<rbmd::Real>(
        send_buffer,
        send_position_offset + static_cast<size_t>(send_count) *
                                   sizeof(rbmd::Real) * 2,
        send_count);
    pending.shift_x.resize(static_cast<size_t>(send_count));
    pending.shift_y.resize(static_cast<size_t>(send_count));
    pending.shift_z.resize(static_cast<size_t>(send_count));
    pending.send_reference_x.resize(static_cast<size_t>(send_count));
    pending.send_reference_y.resize(static_cast<size_t>(send_count));
    pending.send_reference_z.resize(static_cast<size_t>(send_count));
    for (int j = 0; j < send_count; ++j) {
      const auto found = local_indices.find(pending.send_ids[j]);
      if (found == local_indices.end() || found->second.empty()) {
        _pending_forward_valid = false;
        return;
      }
      size_t source = found->second.front();
      rbmd::Real best_distance = std::numeric_limits<rbmd::Real>::max();
      for (const size_t candidate : found->second) {
        const rbmd::Real dx = send_px[j] - px[candidate];
        const rbmd::Real dy = send_py[j] - py[candidate];
        const rbmd::Real dz = send_pz[j] - pz[candidate];
        const rbmd::Real distance = dx * dx + dy * dy + dz * dz;
        if (distance < best_distance) {
          source = candidate;
          best_distance = distance;
        }
      }
      pending.shift_x[j] = send_px[j] - px[source];
      pending.shift_y[j] = send_py[j] - py[source];
      pending.shift_z[j] = send_pz[j] - pz[source];
      pending.send_reference_x[j] = px[source];
      pending.send_reference_y[j] = py[source];
      pending.send_reference_z[j] = pz[source];
    }

    const char* recv_buffer = partner.GetRecvBufferPtr(HALO_COPIES);
    pending.recv_ids =
        CopyDeviceBufferColumn<rbmd::Id>(recv_buffer, 0, recv_count);
    const size_t recv_position_offset =
        static_cast<size_t>(recv_count) * sizeof(rbmd::Id) * 2;
    pending.recv_reference_x = CopyDeviceBufferColumn<rbmd::Real>(
        recv_buffer, recv_position_offset, recv_count);
    pending.recv_reference_y = CopyDeviceBufferColumn<rbmd::Real>(
        recv_buffer,
        recv_position_offset + static_cast<size_t>(recv_count) *
                                   sizeof(rbmd::Real),
        recv_count);
    pending.recv_reference_z = CopyDeviceBufferColumn<rbmd::Real>(
        recv_buffer,
        recv_position_offset + static_cast<size_t>(recv_count) *
                                   sizeof(rbmd::Real) * 2,
        recv_count);
    pending.recv_retained.assign(static_cast<size_t>(recv_count), 0);
  }
}

void IndirectNeighbourCommunicationScheme::MarkRetainedGhosts(
    unsigned short d, int partner_index, rbmd::Id old_total,
    rbmd::Id new_total) {
  if (!_capture_forward_coordinates || !_pending_forward_valid ||
      d >= _pending_forward_stages.size() || partner_index < 0 ||
      static_cast<size_t>(partner_index) >=
          _pending_forward_stages[d].size() ||
      new_total <= old_total || !_device_data) {
    return;
  }

  auto& pending = _pending_forward_stages[d][partner_index];
  thrust::host_vector<rbmd::Id> inserted(
      static_cast<size_t>(new_total - old_total));
  thrust::copy(_device_data->_d_atoms_id.begin() + old_total,
               _device_data->_d_atoms_id.begin() + new_total,
               inserted.begin());
  std::unordered_map<rbmd::Id, int> remaining;
  remaining.reserve(inserted.size());
  for (const auto id : inserted) {
    ++remaining[id];
  }
  for (size_t i = 0; i < pending.recv_ids.size(); ++i) {
    auto found = remaining.find(pending.recv_ids[i]);
    if (found != remaining.end() && found->second > 0) {
      pending.recv_retained[i] = 1;
      --found->second;
    }
  }
}

bool IndirectNeighbourCommunicationScheme::PrepareGhostCoordinateExchange(
    LinkedCell* moleculeContainer) {
  InvalidateGhostCoordinateExchange();
  if (!_capture_forward_coordinates || !_pending_forward_valid ||
      moleculeContainer == nullptr || !_device_data) {
    return false;
  }

  const rbmd::Id total_atoms = moleculeContainer->_total_atoms_num;
  if (total_atoms < 0 ||
      _device_data->_d_atoms_id.size() < static_cast<size_t>(total_atoms)) {
    return false;
  }
  thrust::host_vector<rbmd::Id> ids(static_cast<size_t>(total_atoms));
  thrust::host_vector<rbmd::Real> px(static_cast<size_t>(total_atoms));
  thrust::host_vector<rbmd::Real> py(static_cast<size_t>(total_atoms));
  thrust::host_vector<rbmd::Real> pz(static_cast<size_t>(total_atoms));
  thrust::copy_n(_device_data->_d_atoms_id.begin(), total_atoms, ids.begin());
  thrust::copy_n(_device_data->_d_px.begin(), total_atoms, px.begin());
  thrust::copy_n(_device_data->_d_py.begin(), total_atoms, py.begin());
  thrust::copy_n(_device_data->_d_pz.begin(), total_atoms, pz.begin());
  std::unordered_map<rbmd::Id, std::vector<rbmd::Id>> indices_by_id;
  indices_by_id.reserve(static_cast<size_t>(total_atoms));
  for (rbmd::Id i = 0; i < total_atoms; ++i) {
    indices_by_id[ids[static_cast<size_t>(i)]].push_back(i);
  }

  const auto closest_index = [&](rbmd::Id gid, rbmd::Real ref_x,
                                 rbmd::Real ref_y, rbmd::Real ref_z) {
    const auto found = indices_by_id.find(gid);
    if (found == indices_by_id.end() || found->second.empty()) {
      return rbmd::Id(-1);
    }
    rbmd::Id best = found->second.front();
    rbmd::Real best_distance = std::numeric_limits<rbmd::Real>::max();
    for (const rbmd::Id candidate : found->second) {
      const std::size_t idx = static_cast<std::size_t>(candidate);
      const rbmd::Real dx = px[idx] - ref_x;
      const rbmd::Real dy = py[idx] - ref_y;
      const rbmd::Real dz = pz[idx] - ref_z;
      const rbmd::Real distance = dx * dx + dy * dy + dz * dz;
      if (distance < best_distance ||
          (distance == best_distance && candidate < best)) {
        best = candidate;
        best_distance = distance;
      }
    }
    return best;
  };

  _forward_stages.resize(_pending_forward_stages.size());
  for (size_t d = 0; d < _pending_forward_stages.size(); ++d) {
    auto& target_stage = _forward_stages[d];
    for (const auto& pending : _pending_forward_stages[d]) {
      if (pending.send_ids.empty() && pending.recv_ids.empty()) {
        continue;
      }
      ForwardPeer peer;
      peer.rank = pending.rank;
      peer.receive_side = pending.receive_side;
      std::vector<rbmd::Id> send_indices(pending.send_ids.size());
      for (size_t i = 0; i < pending.send_ids.size(); ++i) {
        if (i >= pending.send_reference_x.size() ||
            i >= pending.send_reference_y.size() ||
            i >= pending.send_reference_z.size()) {
          InvalidateGhostCoordinateExchange();
          return false;
        }
        send_indices[i] = closest_index(
            pending.send_ids[i], pending.send_reference_x[i],
            pending.send_reference_y[i], pending.send_reference_z[i]);
        if (send_indices[i] < 0) {
          InvalidateGhostCoordinateExchange();
          return false;
        }
      }
      std::vector<rbmd::Id> recv_indices(pending.recv_ids.size(), -1);
      for (size_t i = 0; i < pending.recv_ids.size(); ++i) {
        if (i >= pending.recv_retained.size() || !pending.recv_retained[i]) {
          continue;
        }
        if (i >= pending.recv_reference_x.size() ||
            i >= pending.recv_reference_y.size() ||
            i >= pending.recv_reference_z.size()) {
          InvalidateGhostCoordinateExchange();
          return false;
        }
        recv_indices[i] = closest_index(
            pending.recv_ids[i], pending.recv_reference_x[i],
            pending.recv_reference_y[i], pending.recv_reference_z[i]);
        if (recv_indices[i] < 0) {
          InvalidateGhostCoordinateExchange();
          return false;
        }
      }
      if (send_indices.size() >
              static_cast<size_t>(std::numeric_limits<int>::max() / 3) ||
          recv_indices.size() >
              static_cast<size_t>(std::numeric_limits<int>::max() / 3)) {
        InvalidateGhostCoordinateExchange();
        return false;
      }
      peer.send_indices = thrust::device_vector<rbmd::Id>(
          send_indices.begin(), send_indices.end());
      peer.recv_indices = thrust::device_vector<rbmd::Id>(
          recv_indices.begin(), recv_indices.end());
      peer.shift_x = thrust::device_vector<rbmd::Real>(
          pending.shift_x.begin(), pending.shift_x.end());
      peer.shift_y = thrust::device_vector<rbmd::Real>(
          pending.shift_y.begin(), pending.shift_y.end());
      peer.shift_z = thrust::device_vector<rbmd::Real>(
          pending.shift_z.begin(), pending.shift_z.end());
      target_stage.emplace_back(std::move(peer));
    }
  }
  _forward_coordinates_valid = true;
  return true;
}

void IndirectNeighbourCommunicationScheme::ForwardGhostCoordinates(
    LinkedCell* moleculeContainer, DomainDecomposition* domainDecomp) {
  ForwardGhostFields(ForwardGhostState::Coordinates, moleculeContainer,
                     domainDecomp);
}

void IndirectNeighbourCommunicationScheme::ForwardGhostFields(
    ForwardGhostState state, LinkedCell* moleculeContainer,
    DomainDecomposition* domainDecomp) {
  if (!_forward_coordinates_valid || moleculeContainer == nullptr ||
      domainDecomp == nullptr || !_device_data) {
    throw std::runtime_error(
        "Three-stage forward ghost exchange plan is not valid");
  }

  const ForwardFieldLayout layout = MakeForwardFieldLayout(state, _device_data);
  constexpr int kForwardCoordinateTagBase = 29400;
  for (unsigned short d = 0; d < GetCommDims(); ++d) {
    auto& stage = _forward_stages[d];
    std::vector<MPI_Request> requests;
    requests.reserve(stage.size() * 2);
    for (auto& peer : stage) {
      const std::size_t recv_count = peer.recv_indices.size();
      if (recv_count >
          static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                   layout.count)) {
        throw std::overflow_error("Forward ghost receive bundle is too large");
      }
      peer.recv_values.resize(recv_count *
                              static_cast<std::size_t>(layout.count));
      if (peer.recv_values.empty()) {
        continue;
      }
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_CHECK(MPI_Irecv(thrust::raw_pointer_cast(peer.recv_values.data()),
                          static_cast<int>(peer.recv_values.size()),
                          MPI_RBMD_REAL, peer.rank,
                          kForwardCoordinateTagBase +
                              2 * static_cast<int>(d) + peer.receive_side,
                          domainDecomp->_mpi_comm, &request));
      requests.push_back(request);
    }

    for (auto& peer : stage) {
      const rbmd::Id count =
          static_cast<rbmd::Id>(peer.send_indices.size());
      if (count <= 0) {
        peer.send_values.clear();
        continue;
      }
      if (peer.send_indices.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                   layout.count)) {
        throw std::overflow_error("Forward ghost send bundle is too large");
      }
      peer.send_values.resize(peer.send_indices.size() *
                              static_cast<std::size_t>(layout.count));
      PackForwardFields pack{
          thrust::raw_pointer_cast(peer.send_indices.data()),
          thrust::raw_pointer_cast(peer.shift_x.data()),
          thrust::raw_pointer_cast(peer.shift_y.data()),
          thrust::raw_pointer_cast(peer.shift_z.data()),
          {},
          {},
          thrust::raw_pointer_cast(peer.send_values.data()),
          count,
          layout.count};
      for (int field = 0; field < layout.count; ++field) {
        pack.source[field] = layout.source[field];
        pack.shift_dimension[field] = layout.shift_dimension[field];
      }
      thrust::for_each(
          thrust::device, thrust::make_counting_iterator<rbmd::Id>(0),
          thrust::make_counting_iterator<rbmd::Id>(count),
          pack);
    }
    CHECK_RUNTIME(DEVICESYNC());

    for (auto& peer : stage) {
      if (peer.send_values.empty()) {
        continue;
      }
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_CHECK(MPI_Isend(thrust::raw_pointer_cast(peer.send_values.data()),
                          static_cast<int>(peer.send_values.size()),
                          MPI_RBMD_REAL, peer.rank,
                          kForwardCoordinateTagBase +
                              2 * static_cast<int>(d) +
                              OppositeReceiveSide(peer.receive_side),
                          domainDecomp->_mpi_comm, &request));
      requests.push_back(request);
    }
    if (!requests.empty()) {
      MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                            MPI_STATUSES_IGNORE));
    }
    CHECK_RUNTIME(DEVICESYNC());

    for (auto& peer : stage) {
      const rbmd::Id count =
          static_cast<rbmd::Id>(peer.recv_indices.size());
      if (count <= 0) {
        continue;
      }
      ApplyForwardFields apply{
          thrust::raw_pointer_cast(peer.recv_values.data()),
          thrust::raw_pointer_cast(peer.recv_indices.data()),
          {},
          count,
          layout.count};
      for (int field = 0; field < layout.count; ++field) {
        apply.destination[field] = layout.destination[field];
      }
      thrust::for_each(
          thrust::device, thrust::make_counting_iterator<rbmd::Id>(0),
          thrust::make_counting_iterator<rbmd::Id>(count),
          apply);
    }
    // 下一维可能立即转发本维刚更新的 ghost，因此维度之间必须完成同步。
    CHECK_RUNTIME(DEVICESYNC());
  }
}

void IndirectNeighbourCommunicationScheme::packPhaseBuffers(
    unsigned short d, Phase phase, MessageType msgType,
    LinkedCell* moleculeContainer, DomainDecomposition* domainDecomp,
    std::vector<int>& sendCounts, std::vector<int>& sendBytes) {
  auto& neighbours = (*_neighbours)[d];
  const int neighbourCount = static_cast<int>(neighbours.size());
  sendCounts.assign(neighbourCount, 0);
  sendBytes.assign(neighbourCount, 0);

  if (neighbourCount == 0) {
    return;
  }

  rbmd::Id end_idx = 0;
  if (msgType == HALO_COPIES) {
    // 对于HALO_COPIES，需要包括之前阶段接收的所有halo
    // LinkedCell::_total_atoms_num 在 ProcessGhostData 后已经包含新halo
    end_idx = moleculeContainer->_total_atoms_num;
  } else if (msgType == LEAVING_ONLY) {
    end_idx = moleculeContainer->_native_atoms_num;
  }

  // staged mask 只用于 leaving 阶段的“延迟删除”账本。
  // Halo 阶段去重由 CommunicationPartner 内部基于“本 peer 实际 leaving 发送列表”
  // 完成，避免跨 peer 误删 halo。
  thrust::device_vector<int>* staged_mask =
      (msgType == LEAVING_ONLY || msgType == LEAVING_AND_HALO_COPIES)
          ? &_staged_leaving_marks
          : nullptr;

  for (int i = 0; i < neighbourCount; ++i) {
    auto& partner = neighbours[i];
    if (domainDecomp->_current_rank == partner.getRank()) {
      continue;
    }
    
    partner.PrepareMessage(moleculeContainer, domainDecomp->_h_global_box.get(),
                           leaving_flags, halo_flags, msgType, domainDecomp->_mpi_comm,
                           domainDecomp->_current_rank, domainDecomp->_total_ranks,
                           0, end_idx, staged_mask);
    sendCounts[i] = partner.GetSendCount(msgType);
    const auto bytes = partner.GetSendBytes(msgType);
    if (bytes > std::numeric_limits<int>::max()) {
      throw std::runtime_error("MPI 发送缓冲区超过 int 上限，需调整协议实现。");
    }
    sendBytes[i] = static_cast<int>(bytes);
  }
}

void IndirectNeighbourCommunicationScheme::allocateRecvBuffers(
    unsigned short d, Phase phase, const std::vector<int>& recvBytes) {
  auto& neighbours = (*_neighbours)[d];
  MessageType msgType = PhaseToMessageType(phase);
  for (int i = 0; i < static_cast<int>(neighbours.size()); ++i) {
    auto& partner = neighbours[i];
    if (recvBytes[i] <= 0) {
      partner.EnsureRecvBuffer(msgType, 0);
      continue;
    }
    partner.EnsureRecvBuffer(msgType, static_cast<size_t>(recvBytes[i]));
  }
}

IndirectNeighbourCommunicationScheme::PhaseExchangeResult
IndirectNeighbourCommunicationScheme::exchangePhaseP2P(
    unsigned short d, Phase phase, MessageType msgType,
    LinkedCell* moleculeContainer, DomainDecomposition* domainDecomp,
    std::vector<int>* tracked_send_gid_counts,
    std::vector<std::vector<std::pair<int, int>>>* tracked_send_gid_peers) {
  PhaseExchangeResult result{};
  auto& neighbours = (*_neighbours)[d];
  const int neighbourCount = static_cast<int>(neighbours.size());
  if (neighbourCount == 0) {
    return result;
  }

  static bool tagChecked = false;
  if (!tagChecked) {
    int flag = 0;
    int* tagUbPtr = nullptr;
    MPI_Comm_get_attr(domainDecomp->_mpi_comm, MPI_TAG_UB, &tagUbPtr, &flag);
    if (flag && tagUbPtr != nullptr) {
      const int maxTagCandidate =
          _tags.PayloadTag(GetCommDims() - 1, Phase::Halo, 1, 1);
      if (maxTagCandidate > *tagUbPtr) {
        std::ostringstream oss;
        oss << "MPI tag 上限不足，所需 tag=" << maxTagCandidate
            << " 而 MPI_TAG_UB=" << *tagUbPtr
            << "，请调整 MsgTags 配置或 MPI 设置。";
        throw std::runtime_error(oss.str());
      }
    }
    tagChecked = true;
  }

  std::vector<int> sendBytes;
  packPhaseBuffers(d, phase, msgType, moleculeContainer, domainDecomp,
                   result.send_counts, sendBytes);
  result.send_bytes = sendBytes;

  const auto& route_cfg = GetExchangeRouteTraceConfig();
  const bool trace_route = ShouldTraceExchangeRoute(test_current_step);
  const bool collect_tracked_send_counts =
      tracked_send_gid_counts != nullptr && !route_cfg.gids.empty();
  const bool collect_tracked_send_peers =
      tracked_send_gid_peers != nullptr && !route_cfg.gids.empty();
  if (tracked_send_gid_counts != nullptr) {
    tracked_send_gid_counts->assign(route_cfg.gids.size(), 0);
  }
  if (tracked_send_gid_peers != nullptr &&
      tracked_send_gid_peers->size() < route_cfg.gids.size()) {
    tracked_send_gid_peers->resize(route_cfg.gids.size());
  }
  const bool leaving_phase = (phase == Phase::Leaving);

  auto extract_partner_gid_counts = [&](bool is_send, int partner_idx,
                                        int atom_count,
                                        int byte_count) -> std::vector<int> {
    std::vector<int> gid_counts(route_cfg.gids.size(), 0);
    if ((!trace_route && !collect_tracked_send_counts &&
         !collect_tracked_send_peers) ||
        route_cfg.gids.empty()) {
      return gid_counts;
    }
    if (partner_idx < 0 || partner_idx >= neighbourCount) {
      return gid_counts;
    }
    auto& partner = neighbours[partner_idx];
    if (domainDecomp->_current_rank == partner.getRank()) {
      return gid_counts;
    }

    const char* ptr = is_send ? partner.GetSendBufferPtr(msgType)
                              : partner.GetRecvBufferPtr(msgType);
    if (ptr && atom_count > 0 && byte_count > 0) {
      gid_counts = (msgType == LEAVING_ONLY || msgType == LEAVING_AND_HALO_COPIES)
                       ? ExtractTrackedCountsFromLeavingBuffer(
                             ptr, byte_count, atom_count, route_cfg.gids)
                       : ExtractTrackedCountsFromHaloBuffer(
                             ptr, byte_count, atom_count, route_cfg.gids);
    }
    return gid_counts;
  };

  if (trace_route || collect_tracked_send_counts || collect_tracked_send_peers) {
    for (int i = 0; i < neighbourCount; ++i) {
      auto& partner = neighbours[i];
      if (domainDecomp->_current_rank == partner.getRank()) {
        continue;
      }
      const int atom_count =
          result.send_counts.size() > static_cast<size_t>(i)
              ? result.send_counts[i]
              : 0;
      const int byte_count =
          sendBytes.size() > static_cast<size_t>(i) ? sendBytes[i] : 0;
      const auto gid_counts =
          extract_partner_gid_counts(true, i, atom_count, byte_count);

      if (trace_route) {
        AppendExchangeRouteRows(route_cfg, test_current_step, _epoch, d,
                                leaving_phase, msgType, "send",
                                domainDecomp->_current_rank, partner.getRank(),
                                atom_count, byte_count, gid_counts);
      }
      if (collect_tracked_send_counts) {
        for (size_t k = 0; k < gid_counts.size(); ++k) {
          (*tracked_send_gid_counts)[k] += gid_counts[k];
        }
      }
      if (collect_tracked_send_peers && leaving_phase) {
        for (size_t k = 0; k < gid_counts.size() &&
                           k < tracked_send_gid_peers->size();
             ++k) {
          const int gid_send_count = gid_counts[k];
          if (gid_send_count <= 0) {
            continue;
          }
          auto& peers_for_gid = (*tracked_send_gid_peers)[k];
          const int peer_rank = partner.getRank();
          const auto it = std::find_if(
              peers_for_gid.begin(), peers_for_gid.end(),
              [peer_rank](const std::pair<int, int>& item) {
                return item.first == peer_rank;
              });
          if (it == peers_for_gid.end()) {
            peers_for_gid.push_back({peer_rank, gid_send_count});
          } else {
            it->second += gid_send_count;
          }
        }
      }
    }
  }

  auto trace_partner_buffer = [&](bool is_send, int partner_idx,
                                  int atom_count, int byte_count) {
    if (!trace_route || route_cfg.gids.empty()) {
      return;
    }
    if (partner_idx < 0 || partner_idx >= neighbourCount) {
      return;
    }
    auto& partner = neighbours[partner_idx];
    if (domainDecomp->_current_rank == partner.getRank()) {
      return;
    }
    const auto gid_counts =
        extract_partner_gid_counts(is_send, partner_idx, atom_count, byte_count);
    AppendExchangeRouteRows(route_cfg, test_current_step, _epoch, d,
                            leaving_phase, msgType,
                            is_send ? "send" : "recv",
                            domainDecomp->_current_rank, partner.getRank(),
                            atom_count, byte_count, gid_counts);
  };

  std::vector<Header> sendHeaders(neighbourCount);
  std::vector<Header> recvHeaders(neighbourCount);
  std::vector<MPI_Request> headerRecvs;
  std::vector<MPI_Request> headerSends;
  headerRecvs.reserve(neighbourCount);
  headerSends.reserve(neighbourCount);

  const int headerTag = _tags.HeaderTag(d, phase, _epoch, 0);
  const int payloadTag = _tags.PayloadTag(d, phase, _epoch, 0);

  // === DEBUG: 打印 tag 信息，验证唯一性 ===
  int current_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

  auto header_tag_key = std::make_tuple(d, phase, _epoch, headerTag, payloadTag);
  if (_used_tags.find(header_tag_key) != _used_tags.end()) {
    std::cerr << "[ERROR] Tag COLLISION detected: d=" << d << ", phase=" << static_cast<int>(phase)
              << ", epoch=" << _epoch << " | headerTag=" << headerTag << ", payloadTag=" << payloadTag << std::endl;
  }
  _used_tags.insert(header_tag_key);

  // std::cout << "[DEBUG-TAG] Rank " << current_rank << " | Phase=" << (phase == Phase::Leaving ? "Leaving" : "Halo")
  //           << " | Dim=" << d << " | Epoch=" << _epoch
  //           << " | HeaderTag=" << headerTag << " | PayloadTag=" << payloadTag
  //           << " | Neighbours=" << neighbourCount << std::endl;

  for (int i = 0; i < neighbourCount; ++i) {
    auto& partner = neighbours[i];
    if (domainDecomp->_current_rank == partner.getRank()) {
      continue;
    }
    MPI_Request req{};
    const int receive_side = PartnerReceiveSide(partner, d);
    MPI_Irecv(reinterpret_cast<char*>(&recvHeaders[i]), sizeof(Header),
              MPI_BYTE, partner.getRank(),
              _tags.HeaderTag(d, phase, _epoch, receive_side),
              domainDecomp->_mpi_comm, &req);
    headerRecvs.push_back(req);
  }

  for (int i = 0; i < neighbourCount; ++i) {
    auto& partner = neighbours[i];
    sendHeaders[i].count = result.send_counts.size() > static_cast<size_t>(i)
                               ? result.send_counts[i]
                               : 0;
    sendHeaders[i].nbytes =
        sendBytes.size() > static_cast<size_t>(i) ? sendBytes[i] : 0;
    sendHeaders[i].epoch = _epoch;
    if (domainDecomp->_current_rank == partner.getRank()) {
      continue;
    }
    MPI_Request req{};
    const int send_side =
        OppositeReceiveSide(PartnerReceiveSide(partner, d));
    MPI_Isend(reinterpret_cast<char*>(&sendHeaders[i]), sizeof(Header),
              MPI_BYTE, partner.getRank(),
              _tags.HeaderTag(d, phase, _epoch, send_side),
              domainDecomp->_mpi_comm, &req);
    headerSends.push_back(req);
  }

  if (!headerRecvs.empty()) {
    MPI_Waitall(static_cast<int>(headerRecvs.size()), headerRecvs.data(),
                MPI_STATUSES_IGNORE);
  }
  if (!headerSends.empty()) {
    MPI_Waitall(static_cast<int>(headerSends.size()), headerSends.data(),
                MPI_STATUSES_IGNORE);
  }

  result.recv_counts.assign(neighbourCount, 0);
  std::vector<int> recvBytes(neighbourCount, 0);
  for (int i = 0; i < neighbourCount; ++i) {
    result.recv_counts[i] = recvHeaders[i].count;
    recvBytes[i] = recvHeaders[i].nbytes;
  }
  result.recv_bytes = recvBytes;

  allocateRecvBuffers(d, phase, recvBytes);

  std::vector<MPI_Request> payloadRecvs;
  std::vector<MPI_Request> payloadSends;
  payloadRecvs.reserve(neighbourCount);
  payloadSends.reserve(neighbourCount);

  for (int i = 0; i < neighbourCount; ++i) {
    auto& partner = neighbours[i];
    if (domainDecomp->_current_rank == partner.getRank()) {
      continue;
    }
    const int bytes = recvBytes[i];
    if (bytes <= 0) {
      continue;
    }
    char* recvPtr = partner.GetRecvBufferPtr(msgType);
    MPI_Request req{};
    const int receive_side = PartnerReceiveSide(partner, d);
    MPI_Irecv(recvPtr, bytes, MPI_BYTE, partner.getRank(),
              _tags.PayloadTag(d, phase, _epoch, receive_side),
              domainDecomp->_mpi_comm, &req);
    payloadRecvs.push_back(req);
  }

  for (int i = 0; i < neighbourCount; ++i) {
    auto& partner = neighbours[i];
    if (domainDecomp->_current_rank == partner.getRank()) {
      continue;
    }
    const int bytes = sendBytes[i];
    if (bytes <= 0) {
      partner.ClearSendBuffer();
      continue;
    }
    char* sendPtr = partner.GetSendBufferPtr(msgType);
    MPI_Request req{};
    const int send_side =
        OppositeReceiveSide(PartnerReceiveSide(partner, d));
    MPI_Isend(sendPtr, bytes, MPI_BYTE, partner.getRank(),
              _tags.PayloadTag(d, phase, _epoch, send_side),
              domainDecomp->_mpi_comm, &req);
    payloadSends.push_back(req);
  }

  if (!payloadRecvs.empty()) {
    MPI_Waitall(static_cast<int>(payloadRecvs.size()), payloadRecvs.data(),
                MPI_STATUSES_IGNORE);
  }
  if (!payloadSends.empty()) {
    MPI_Waitall(static_cast<int>(payloadSends.size()), payloadSends.data(),
                MPI_STATUSES_IGNORE);
  }

  if (trace_route) {
    for (int i = 0; i < neighbourCount; ++i) {
      const int atom_count =
          result.recv_counts.size() > static_cast<size_t>(i)
              ? result.recv_counts[i]
              : 0;
      const int byte_count =
          recvBytes.size() > static_cast<size_t>(i) ? recvBytes[i] : 0;
      trace_partner_buffer(false, i, atom_count, byte_count);
    }
  }

  if (phase == Phase::Halo) {
    CaptureGhostPhase(d, result, moleculeContainer, domainDecomp);
  }

  for (int i = 0; i < neighbourCount; ++i) {
    neighbours[i].ClearSendBuffer();
  }

  return result;
}

void IndirectNeighbourCommunicationScheme::convert1StageTo3StageNeighbours(

    const std::vector<CommunicationPartner>& commPartners,
    std::vector<std::vector<CommunicationPartner> >& neighbours,
    rbmd::Real cutoffRadius) {
  // TODO: extend for anything else than full shell ?
  // TODO: implement conversion of 1StageTo3StageNeighbours

  int current_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

  // 首先，按维度分组所有面邻居
  std::map<unsigned int, std::vector<CommunicationPartner>> dim_partners;

  for (const CommunicationPartner& commPartner : commPartners) {
    if (!commPartner.isFaceCommunicator()) {
      continue;
      // if commPartner is not a face sharing communicator, we can ignore it!
    }
    unsigned int d = commPartner.getFaceCommunicationDirection();
    dim_partners[d].push_back(commPartner);
  }

  // 现在为每个维度添加邻居
  for (unsigned int d = 0; d < 3; ++d) {
    if (dim_partners.find(d) == dim_partners.end()) {
      // 当该维度进程网格尺寸为 1 时（coversWholeDomain），该维度不需要 MPI 邻居通信。
      if (!_coversWholeDomain[d]) {
        std::cerr << "[ERROR] Rank " << current_rank
                  << " | Dimension " << d << " has NO neighbours at all!" << std::endl;
      }
      continue;
    }

    auto& partners_in_d = dim_partners[d];
    const rbmd::Real local_extent =
        _h_local_box->_coord_max[d] - _h_local_box->_coord_min[d];
    const std::array<rbmd::Real, 3> local_extents{
        _h_local_box->_coord_max[0] - _h_local_box->_coord_min[0],
        _h_local_box->_coord_max[1] - _h_local_box->_coord_min[1],
        _h_local_box->_coord_max[2] - _h_local_box->_coord_min[2],
    };

    for (const auto& commPartner : partners_in_d) {
      neighbours[d].push_back(commPartner);
      neighbours[d].back().enlargeInOtherDirections(d, cutoffRadius);
      neighbours[d].back().ExtendLeavingAlongFaceDirection(d, local_extent);
      neighbours[d].back().ExtendLeavingInOtherDirections(d, local_extents);
    }

    // 如果只有1个邻居，这是一个警告（在周期边界条件下应该有2个）
    if (partners_in_d.size() == 1) {
      std::cerr << "[WARN] Rank " << current_rank
                << " | Dimension " << d << " has only 1 neighbour (expected 2)!" << std::endl;
    }
  }
}

void IndirectNeighbourCommunicationScheme::ValidateGhostCutoffState(
    const LinkedCell* linked_cell, const char* context) const {
  if (linked_cell == nullptr) {
    std::ostringstream oss;
    oss << context << " requires a valid linked_cell";
    throw std::runtime_error(oss.str());
  }

  const auto communication_cutoff =
      static_cast<rbmd::Real>(_ghost_cutoff);
  rbmd::neighbor::ValidateHaloCutoffCoversNeighborCutoff(
      linked_cell->_halo_cutoff, linked_cell->_neighbor_cutoff, context);
  rbmd::neighbor::ValidateHaloCutoffCoversNeighborCutoff(
      communication_cutoff, linked_cell->_neighbor_cutoff, context);

  if (!NearlyEqualCutoff(communication_cutoff, linked_cell->_halo_cutoff)) {
    std::ostringstream oss;
    oss << context
        << " communication cutoff and linked-cell halo cutoff diverged:"
        << " communication_cutoff=" << communication_cutoff
        << ", linked_cell_halo_cutoff=" << linked_cell->_halo_cutoff
        << ", neighbor_cutoff=" << linked_cell->_neighbor_cutoff;
    throw std::runtime_error(oss.str());
  }
}

void IndirectNeighbourCommunicationScheme::exchangeMoleculesMPI(
    LinkedCell* moleculeContainer, DomainDecomposition* domainDecomp) {
  ValidateGhostCutoffState(moleculeContainer,
                           "IndirectNeighbourCommunicationScheme::exchange");
  ++_epoch;
  InvalidateGhostCoordinateExchange();
  _capture_forward_coordinates =
      moleculeContainer != nullptr && moleculeContainer->_skin > rbmd::Real(0);
  _pending_forward_valid = _capture_forward_coordinates;
  _pending_forward_stages.clear();
  _pending_forward_stages.resize(GetCommDims());
  const rbmd::Id total_atoms = _linked_cell->_total_atoms_num;
  _staged_leaving_marks.resize(static_cast<size_t>(total_atoms));
  thrust::fill(_staged_leaving_marks.begin(), _staged_leaving_marks.end(), 0);
  const auto& route_cfg = GetExchangeRouteTraceConfig();
  const bool trace_stage_stats =
      ShouldTraceExchangeStageStats(test_current_step);
  const bool check_leaving_consistency =
      ShouldCheckLeavingConsistency(test_current_step);
  const bool has_tracked_gids = !route_cfg.gids.empty();
  std::vector<int> leaving_sent_gid_counts(route_cfg.gids.size(), 0);
  std::vector<std::vector<std::pair<int, int>>> leaving_sent_gid_peers(
      route_cfg.gids.size());
  rbmd::Id leaving_sent_total = 0;
  bool has_p2p_leaving_phase = false;
  rbmd::Id leaving_send_atoms = 0;
  rbmd::Id leaving_recv_atoms = 0;
  long long leaving_send_bytes = 0;
  long long leaving_recv_bytes = 0;
  rbmd::Id halo_send_atoms = 0;
  rbmd::Id halo_recv_atoms = 0;
  long long halo_send_bytes = 0;
  long long halo_recv_bytes = 0;

  auto sum_int_vector = [](const std::vector<int>& values) -> long long {
    long long total_value = 0;
    for (const int value : values) {
      if (value > 0) {
        total_value += static_cast<long long>(value);
      }
    }
    return total_value;
  };

  if (trace_stage_stats) {
    AppendExchangeStageStatsRow("before_leaving", test_current_step, _epoch,
                                domainDecomp->_current_rank,
                                _linked_cell->_native_atoms_num,
                                _linked_cell->_ghost_atoms_num,
                                _linked_cell->_total_atoms_num, 0, 0, 0, 0);
  }


  // ========== 阶段 1: 全维度处理 Leaving 粒子 ==========
  for (unsigned short d = 0; d < GetCommDims(); ++d) {
    if (_coversWholeDomain[d]) {
      domainDecomp->HandleDomainLeavingAtoms(d, moleculeContainer);
    } else {
      has_p2p_leaving_phase = true;
      std::vector<int> dim_tracked_send_counts;
      auto phaseResult = exchangePhaseP2P(d, Phase::Leaving, LEAVING_ONLY,
                                          moleculeContainer, domainDecomp,
                                          (check_leaving_consistency &&
                                           has_tracked_gids)
                                              ? &dim_tracked_send_counts
                                              : nullptr,
                                          (check_leaving_consistency &&
                                           has_tracked_gids)
                                              ? &leaving_sent_gid_peers
                                              : nullptr);
      leaving_send_atoms += static_cast<rbmd::Id>(
          sum_int_vector(phaseResult.send_counts));
      leaving_recv_atoms += static_cast<rbmd::Id>(
          sum_int_vector(phaseResult.recv_counts));
      leaving_send_bytes += sum_int_vector(phaseResult.send_bytes);
      leaving_recv_bytes += sum_int_vector(phaseResult.recv_bytes);
      if (check_leaving_consistency) {
        for (const int sent : phaseResult.send_counts) {
          if (sent > 0) {
            leaving_sent_total += static_cast<rbmd::Id>(sent);
          }
        }
      }
      if (check_leaving_consistency && has_tracked_gids) {
        for (size_t k = 0;
             k < leaving_sent_gid_counts.size() &&
             k < dim_tracked_send_counts.size();
             ++k) {
          leaving_sent_gid_counts[k] += dim_tracked_send_counts[k];
        }
      }
      auto& neighbours = (*_neighbours)[d];
      for (int i = 0; i < static_cast<int>(neighbours.size()); ++i) {
        if (phaseResult.recv_counts[i] <= 0) {
          continue;
        }

        neighbours[i].ProcessLeavingData(moleculeContainer);

      }
    }
  }
  if (trace_stage_stats) {
    AppendExchangeStageStatsRow("after_leaving", test_current_step, _epoch,
                                domainDecomp->_current_rank,
                                _linked_cell->_native_atoms_num,
                                _linked_cell->_ghost_atoms_num,
                                _linked_cell->_total_atoms_num,
                                leaving_send_atoms, leaving_recv_atoms,
                                leaving_send_bytes, leaving_recv_bytes);
  }
  // 在这里统一删除所有leaving粒子

  auto AssertTrackedLeavingConsistency = [&](const std::string& check_point) {
    if (!check_leaving_consistency || !has_p2p_leaving_phase || !_device_data) {
      return;
    }
    const rbmd::Id native_atoms = _linked_cell->_native_atoms_num;
    const rbmd::Id staged_size =
        static_cast<rbmd::Id>(_staged_leaving_marks.size());
    const rbmd::Id marked_span = std::min(native_atoms, staged_size);
    const rbmd::Id marked_total =
        (marked_span > 0)
            ? thrust::reduce(_staged_leaving_marks.begin(),
                             _staged_leaving_marks.begin() + marked_span)
            : rbmd::Id(0);
    const bool aggregate_mismatch = (marked_total != leaving_sent_total);

    std::vector<int> marked_counts;
    std::vector<rbmd::Id> old_indices;
    bool tracked_mismatch = false;
    if (has_tracked_gids) {
      marked_counts = CountTrackedMarkedLeavingAtoms(
          route_cfg.gids, _staged_leaving_marks, _device_data->_d_atoms_id,
          native_atoms);
      tracked_mismatch =
          HasLeavingConsistencyMismatch(marked_counts, leaving_sent_gid_counts);
      if (tracked_mismatch) {
        old_indices = CollectTrackedFirstNativeIndices(
            route_cfg.gids, _device_data->_d_atoms_id, native_atoms);
      }
    }

    if (aggregate_mismatch || tracked_mismatch) {
      DumpLeavingConsistencyContextAndExit(
          route_cfg, test_current_step, _epoch, check_point,
          domainDecomp->_current_rank, _linked_cell->_native_atoms_num,
          _linked_cell->_ghost_atoms_num, _linked_cell->_total_atoms_num,
          marked_total, leaving_sent_total,
          marked_counts, leaving_sent_gid_counts, old_indices,
          leaving_sent_gid_peers);
    }
  };

  if (static_cast<rbmd::Id>(_staged_leaving_marks.size()) <
      _linked_cell->_total_atoms_num) {
    _staged_leaving_marks.resize(
        static_cast<size_t>(_linked_cell->_total_atoms_num), 0);
  }
  // 第一层防线：离开阶段完成后立即对账（防止“标记了删除但未进入sendlist”）。
  AssertTrackedLeavingConsistency("after_leaving_phase");
  if (_device_data) {
    AppendTrackedGidPresenceRows(
        route_cfg, test_current_step, _epoch, "after_leaving_phase",
        domainDecomp->_current_rank, _linked_cell->_native_atoms_num,
        _linked_cell->_ghost_atoms_num, _linked_cell->_total_atoms_num,
        _device_data->_d_atoms_id, domainDecomp->_mpi_comm);
  }

  auto compressDeferredLeaving = [&]() {
    if (_staged_leaving_marks.empty()) {
      return;
    }
    const rbmd::Id native_atoms = _linked_cell->_native_atoms_num;
    if (native_atoms <= 0) {
      thrust::fill(_staged_leaving_marks.begin(),
                   _staged_leaving_marks.end(), 0);
      return;
    }
    const rbmd::Id total_atoms_now = _linked_cell->_total_atoms_num;
    if (static_cast<rbmd::Id>(_staged_leaving_marks.size()) < total_atoms_now) {
      _staged_leaving_marks.resize(static_cast<size_t>(total_atoms_now), 0);
    }
    // 第二层防线：压缩前再次对账，避免“先删后查”导致丢现场。
    AssertTrackedLeavingConsistency("before_compress_deferred_leaving");
    const rbmd::Id marked = thrust::reduce(
        _staged_leaving_marks.begin(),
        _staged_leaving_marks.begin() + native_atoms);
    if (marked == 0) {
      thrust::fill(_staged_leaving_marks.begin(),
                   _staged_leaving_marks.begin() + native_atoms, 0);
      return;
    }

    // 关键点：leaving 压缩会改变 per-atom 数组的“原子索引→数据行”映射。
    // 当前实现只压缩了 id/type/pos/charge 等基础数组，如果不同时压缩 per-atom topology
    // (bond/angle/dihedral/improper/nspecial/special)，会导致拓扑与原子错位，进而产生
    // 错误的 special 排除/缩放，常见症状是多进程下首次迁移后温度/能量发散。
    //
    // 这里先在压缩基础数组之前，构造 native 区间内“保留原子”的旧索引列表（稳定顺序）。
    const rbmd::Id new_native_atoms = native_atoms - marked;
    const rbmd::Id clamped_new_native_atoms =
        (new_native_atoms > 0) ? new_native_atoms : 0;
    thrust::device_vector<int> kept_native_indices;
    kept_native_indices.resize(static_cast<size_t>(clamped_new_native_atoms));
    if (clamped_new_native_atoms > 0) {
      thrust::copy_if(
          thrust::make_counting_iterator(0),
          thrust::make_counting_iterator(
              CheckedIntCount(native_atoms, "kept native_atoms")),
          _staged_leaving_marks.begin(),
          kept_native_indices.begin(),
          IsKeptMark{});
    }

    auto compact_per_atom_topology = [&](rbmd::Id target_native) {
      if (!_device_data || target_native <= 0) {
        return;
      }
      auto& data = *_device_data;
      if (kept_native_indices.empty()) {
        return;
      }

      const int* keep_ptr = thrust::raw_pointer_cast(kept_native_indices.data());

      // gather helper for 1D per-atom arrays: out[i] = in[keep[i]]
      auto gather_rows_1d = [&](auto& vec) {
        using Vec = std::decay_t<decltype(vec)>;
        using T = typename Vec::value_type;
        if (vec.empty()) {
          return;
        }
        thrust::device_vector<T> tmp(static_cast<size_t>(target_native));
        thrust::gather(kept_native_indices.begin(), kept_native_indices.end(),
                       vec.begin(), tmp.begin());
        thrust::copy(tmp.begin(), tmp.end(), vec.begin());
      };

      // gather helper for fixed-width 2D flattened arrays (row-major):
      // out[new_i * width + k] = in[ keep[new_i] * width + k ]
      auto gather_rows_2d_flat = [&](auto& vec, int width) {
        using Vec = std::decay_t<decltype(vec)>;
        using T = typename Vec::value_type;
        if (vec.empty() || width <= 0 || target_native <= 0) {
          return;
        }
        const size_t out_size =
            static_cast<size_t>(target_native) * static_cast<size_t>(width);
        if (out_size == 0) {
          return;
        }

        thrust::device_vector<T> tmp(out_size);
        auto begin = thrust::make_counting_iterator<size_t>(0);
        auto map_it =
            thrust::make_transform_iterator(begin,
                                            RowMajorGatherIndex{keep_ptr, width});
        // 强制走 device 后端，避免 map iterator 在 host fallback 时触发
        // RowMajorGatherIndex 的 host 分支（返回 0）导致错误重排。
        thrust::gather(thrust::device, map_it, map_it + out_size, vec.begin(),
                       tmp.begin());
        thrust::copy(tmp.begin(), tmp.end(), vec.begin());
      };

      // Bond
      if (!data.d_num_bond.empty() && data.bond_per_atom > 0) {
        gather_rows_1d(data.d_num_bond);
        gather_rows_2d_flat(data.d_bond_type, data.bond_per_atom);
        gather_rows_2d_flat(data.d_bond_atom, data.bond_per_atom);
      }
      // Angle
      if (!data.d_num_angle.empty() && data.angle_per_atom > 0) {
        gather_rows_1d(data.d_num_angle);
        gather_rows_2d_flat(data.d_angle_type, data.angle_per_atom);
        gather_rows_2d_flat(data.d_angle_atom1, data.angle_per_atom);
        gather_rows_2d_flat(data.d_angle_atom2, data.angle_per_atom);
        gather_rows_2d_flat(data.d_angle_atom3, data.angle_per_atom);
      }
      // Dihedral
      if (!data.d_num_dihedral.empty() && data.dihedral_per_atom > 0) {
        gather_rows_1d(data.d_num_dihedral);
        gather_rows_2d_flat(data.d_dihedral_type, data.dihedral_per_atom);
        gather_rows_2d_flat(data.d_dihedral_atom1, data.dihedral_per_atom);
        gather_rows_2d_flat(data.d_dihedral_atom2, data.dihedral_per_atom);
        gather_rows_2d_flat(data.d_dihedral_atom3, data.dihedral_per_atom);
        gather_rows_2d_flat(data.d_dihedral_atom4, data.dihedral_per_atom);
      }
      // Improper
      if (!data.d_num_improper.empty() && data.improper_per_atom > 0) {
        gather_rows_1d(data.d_num_improper);
        gather_rows_2d_flat(data.d_improper_type, data.improper_per_atom);
        gather_rows_2d_flat(data.d_improper_atom1, data.improper_per_atom);
        gather_rows_2d_flat(data.d_improper_atom2, data.improper_per_atom);
        gather_rows_2d_flat(data.d_improper_atom3, data.improper_per_atom);
        gather_rows_2d_flat(data.d_improper_atom4, data.improper_per_atom);
      }
      // Special: nspecial 为 [nmax*3]，special 为 [nmax*maxspecial]
      if (!data.d_nspecial.empty()) {
        gather_rows_2d_flat(data.d_nspecial, 3);
      }
      if (!data.d_special.empty() && data.maxspecial > 0) {
        gather_rows_2d_flat(data.d_special, data.maxspecial);
      }
    };

    auto atom_style =
        DataManager::getInstance().getConfigData()->Get<std::string>(
            "atom_style", "init_configuration", "read_data");
    const size_t total = static_cast<size_t>(total_atoms_now);
    const size_t native = static_cast<size_t>(native_atoms);
    auto mask_begin = _staged_leaving_marks.begin();
    if (atom_style == "atomic") {
      auto begin = thrust::make_zip_iterator(thrust::make_tuple(
          _device_data->_d_atoms_id.begin(),
          _device_data->_d_atoms_type.begin(),
          _device_data->_d_px.begin(), _device_data->_d_py.begin(),
          _device_data->_d_pz.begin()));
      auto end = begin + total;
      thrust::remove_if(begin, end, mask_begin,
                                       thrust::identity<int>());
    } else if (atom_style == "charge") {
      auto begin = thrust::make_zip_iterator(thrust::make_tuple(
          _device_data->_d_atoms_id.begin(),
          _device_data->_d_atoms_type.begin(),
          _device_data->_d_px.begin(), _device_data->_d_py.begin(),
          _device_data->_d_pz.begin(), _device_data->_d_charge.begin()));
      auto end = begin + total;
      thrust::remove_if(begin, end, mask_begin,
                                       thrust::identity<int>());
    } else if (atom_style == "full") {
      auto begin = thrust::make_zip_iterator(thrust::make_tuple(
          _device_data->_d_atoms_id.begin(),
          _device_data->_d_atoms_type.begin(),
          _device_data->_d_molecular_id.begin(),
          _device_data->_d_px.begin(), _device_data->_d_py.begin(),
          _device_data->_d_pz.begin(), _device_data->_d_charge.begin()));
      auto end = begin + total;
      thrust::remove_if(begin, end, mask_begin,
                                       thrust::identity<int>());
    } else {
      throw std::runtime_error("Unsupported atom_style for leaving compaction");
    }

    auto native_begin = thrust::make_zip_iterator(thrust::make_tuple(
        _device_data->_d_vx.begin(), _device_data->_d_vy.begin(),
        _device_data->_d_vz.begin()));
    auto native_end = native_begin + native;
    thrust::remove_if(native_begin, native_end,
                      mask_begin, thrust::identity<int>());

    // 压缩 per-atom topology（保持与基础数组相同的稳定顺序）
    compact_per_atom_topology(new_native_atoms);

    // rbsog: 统一通过 ResizeNativeDependentVectors() 更新计数与数组大小。
    // 这里直接设置压缩后的 native 数量，避免与 UpdateLeavingNum()
    // 混用导致计数语义分裂。
    const rbmd::Id updated_native = clamped_new_native_atoms;
    const rbmd::Id updated_total =
        updated_native + _linked_cell->_ghost_atoms_num;
    _linked_cell->ResizeNativeDependentVectors(updated_native, updated_total);
    _staged_leaving_marks.resize(
        static_cast<size_t>(_linked_cell->_total_atoms_num));
    thrust::fill(_staged_leaving_marks.begin(),
                 _staged_leaving_marks.end(), 0);
  };
  compressDeferredLeaving();
  if (_device_data) {
    AppendTrackedGidPresenceRows(
        route_cfg, test_current_step, _epoch, "after_compact_deferred_leaving",
        domainDecomp->_current_rank, _linked_cell->_native_atoms_num,
        _linked_cell->_ghost_atoms_num, _linked_cell->_total_atoms_num,
        _device_data->_d_atoms_id, domainDecomp->_mpi_comm);
  }

  // ========== 阶段 2: 全维度累积 Halo 粒子 ==========
  // Halo sendlists and the reusable forward-coordinate plan must be captured
  // from the final native ordering.  A leaving exchange can migrate atoms and
  // compact their rows, so performing halo capture before this point leaves
  // the cached plan referring to IDs/indices that no longer exist.
  if (trace_stage_stats) {
    AppendExchangeStageStatsRow("before_halo", test_current_step, _epoch,
                                domainDecomp->_current_rank,
                                _linked_cell->_native_atoms_num,
                                _linked_cell->_ghost_atoms_num,
                                _linked_cell->_total_atoms_num, 0, 0, 0, 0);
  }
  for (unsigned short d = 0; d < GetCommDims(); ++d) {
    if (_coversWholeDomain[d]) {
      continue;
    }
    auto phaseResult = exchangePhaseP2P(d, Phase::Halo, HALO_COPIES,
                                        moleculeContainer, domainDecomp);
    halo_send_atoms +=
        static_cast<rbmd::Id>(sum_int_vector(phaseResult.send_counts));
    halo_recv_atoms +=
        static_cast<rbmd::Id>(sum_int_vector(phaseResult.recv_counts));
    halo_send_bytes += sum_int_vector(phaseResult.send_bytes);
    halo_recv_bytes += sum_int_vector(phaseResult.recv_bytes);
    auto& neighbours = (*_neighbours)[d];
    for (int i = 0; i < static_cast<int>(neighbours.size()); ++i) {
      if (phaseResult.recv_counts[i] <= 0) {
        continue;
      }

      const rbmd::Id old_total = moleculeContainer->_total_atoms_num;
      neighbours[i].ProcessGhostData(moleculeContainer);
      MarkRetainedGhosts(d, i, old_total,
                         moleculeContainer->_total_atoms_num);
    }
  }
  if (trace_stage_stats) {
    AppendExchangeStageStatsRow("after_halo", test_current_step, _epoch,
                                domainDecomp->_current_rank,
                                _linked_cell->_native_atoms_num,
                                _linked_cell->_ghost_atoms_num,
                                _linked_cell->_total_atoms_num, halo_send_atoms,
                                halo_recv_atoms, halo_send_bytes,
                                halo_recv_bytes);
  }
  if (_device_data) {
    AppendTrackedGidPresenceRows(
        route_cfg, test_current_step, _epoch, "after_halo_phase",
        domainDecomp->_current_rank, _linked_cell->_native_atoms_num,
        _linked_cell->_ghost_atoms_num, _linked_cell->_total_atoms_num,
        _device_data->_d_atoms_id, domainDecomp->_mpi_comm);
  }

  // ========== 阶段 3: 最终整理 ==========
  _linked_cell->_per_atom_cell_id.resize(_linked_cell->_total_atoms_num);
}

void IndirectNeighbourCommunicationScheme::initCommunicationPartners(
    rbmd::Real cutoffRadius, DomainDecomposition* domainDecomp) {
  _ghost_cutoff = cutoffRadius;
  ValidateGhostCutoffState(_linked_cell.get(),
                           "IndirectNeighbourCommunicationScheme::init");
  for (unsigned int d = 0; d < _commDimms; d++) {
    (*_neighbours)[d].clear();
  }
  // 每个进程的Halo
  HaloRegion ownRegion = {_h_local_box->_coord_min[0],
                          _h_local_box->_coord_min[1],
                          _h_local_box->_coord_min[2],
                          _h_local_box->_coord_max[0],
                          _h_local_box->_coord_max[1],
                          _h_local_box->_coord_max[2],
                          0,
                          0,
                          0,
                          cutoffRadius}; // region of the box  //TODO! !! 这里是否不用cutoff而是用cell的边长好一些呢
  // ---
  std::vector<HaloRegion> haloRegions = _zonalMethod->getLeavingExportRegions(
      ownRegion, cutoffRadius,
      _coversWholeDomain); // halo regions (outside of the box)
  // ---
  std::vector<CommunicationPartner> commPartners;
  for (HaloRegion haloRegion : haloRegions) {
    // determine who to communicate with - who's region is in the haloRegions
    // vector?
    auto newCommPartners =
        domainDecomp->GetNeighboursFromHaloRegion(haloRegion);
    // TODO
    // 原本我想着就是根据posioninfo计算出需要多少空间的，现在的话因为扩充了，enlarge了，所以就不太方便操作。
    commPartners.insert(commPartners.end(), newCommPartners.begin(),
                        newCommPartners.end());
  }
  _fullShellNeighbours = commPartners;
  convert1StageTo3StageNeighbours(commPartners, (*_neighbours), cutoffRadius);
  for (unsigned int d = 0; d < _commDimms; d++) {
    // Keep opposite periodic sides independent even when both map to one rank.
    (*_neighbours)[d] =
        SqueezePartnersByDirection((*_neighbours)[d], d);
  }
}

void IndirectNeighbourCommunicationScheme::prepareNonBlockingStageImpl(
    LinkedCell* moleculeContainer, unsigned int stageNumber,
    MessageType msgType, bool removeRecvDuplicates,
    DomainDecomposition* domainDecomp) {}

void IndirectNeighbourCommunicationScheme::finishNonBlockingStageImpl(
    LinkedCell* moleculeContainer, unsigned int stageNumber,
    MessageType msgType, bool removeRecvDuplicates,
    DomainDecomposition* domainDecomp) {}
