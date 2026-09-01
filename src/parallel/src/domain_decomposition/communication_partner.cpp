#include "communication_partner.h"

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cstdlib>
#include <unistd.h>

#include "common/rbmd_define.h"
#include "data_manager.h"
#include "domain_decomposition_op.h"
#include "domain_decomposition/topology_pack_op.h"
#include "mpi_data_type_utils.h"

// #define NDEBUG  // 注释掉以启用调试输出
extern rbmd::Id test_current_step;


namespace {

int CheckedIntCount(rbmd::Id value, const char* field) {
  if (value > static_cast<rbmd::Id>(std::numeric_limits<int>::max())) {
    std::ostringstream oss;
    oss << field << " exceeds int range required by topology pack kernels: "
        << value;
    throw std::overflow_error(oss.str());
  }
  return static_cast<int>(value);
}

bool EnvFlagEnabled(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool DedupGhostAtomsEnabled() {
  const char* raw = std::getenv("RBMD_DEDUP_GHOST_ATOMS");
  if (raw == nullptr) {
    return true;
  }
  return EnvFlagEnabled("RBMD_DEDUP_GHOST_ATOMS");
}

template <typename T>
thrust::host_vector<T> CopyGhostColumnToHost(char* recv_buffer,
                                             size_t offset,
                                             unsigned int count) {
  thrust::host_vector<T> values(count);
  if (count == 0) {
    return values;
  }
  auto source_raw_ptr = reinterpret_cast<T*>(recv_buffer + offset);
  thrust::device_ptr<T> source_begin_iter(source_raw_ptr);
  thrust::copy(source_begin_iter, source_begin_iter + count, values.begin());
  return values;
}

template <typename T>
thrust::host_vector<T> GatherHostByIndex(
    const thrust::host_vector<T>& source,
    const std::vector<unsigned int>& indices) {
  thrust::host_vector<T> values(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    values[i] = source[indices[i]];
  }
  return values;
}

std::vector<rbmd::Id> ParseTrackedGidsFromEnv() {
  std::vector<rbmd::Id> gids;
  const char* raw = std::getenv("RBMD_DEBUG_TRACK_GIDS");
  if (raw == nullptr || *raw == '\0') {
    return gids;
  }

  std::string normalized(raw);
  for (char& c : normalized) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0 ||
        c == ';' || c == '|') {
      c = ',';
    }
  }

  std::stringstream ss(normalized);
  std::string token;
  while (std::getline(ss, token, ',')) {
    size_t b = 0;
    while (b < token.size() &&
           std::isspace(static_cast<unsigned char>(token[b])) != 0) {
      ++b;
    }
    size_t e = token.size();
    while (e > b &&
           std::isspace(static_cast<unsigned char>(token[e - 1])) != 0) {
      --e;
    }
    if (b >= e) {
      continue;
    }
    const std::string trimmed = token.substr(b, e - b);
    char* endptr = nullptr;
    const long long parsed = std::strtoll(trimmed.c_str(), &endptr, 10);
    if (endptr == trimmed.c_str() || *endptr != '\0') {
      continue;
    }
    gids.push_back(static_cast<rbmd::Id>(parsed));
  }
  return gids;
}

const std::vector<rbmd::Id>& GetHaloDecisionTrackedGids() {
  static const std::vector<rbmd::Id> gids = ParseTrackedGidsFromEnv();
  return gids;
}

bool ShouldTraceHaloDecision() {
  static const bool enabled =
      EnvFlagEnabled("RBMD_DEBUG_EXCHANGE_ROUTE") &&
      !GetHaloDecisionTrackedGids().empty();
  return enabled;
}

bool UsesTopologyExchangeLayout(const std::string& atom_style) {
  return atom_style == "full";
}

const char* RegionName(rbmd::Id idx, rbmd::Id native_atoms, rbmd::Id total_atoms) {
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

bool HostIsInRegion(const rbmd::Real start_region[3],
                    const rbmd::Real end_region[3],
                    rbmd::Real px, rbmd::Real py, rbmd::Real pz) {
  return (px >= start_region[0] && px < end_region[0] &&
          py >= start_region[1] && py < end_region[1] &&
          pz >= start_region[2] && pz < end_region[2]);
}

void EnsureCommunicationDebugLogDirectory() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  (void)mkdir("logs", 0755);
  (void)mkdir("logs/debug", 0755);
  initialized = true;
}

void AppendTrackedNativeCellStateRows(
    rbmd::Id step, int rank, int peer_rank, Box* box, LinkedCell* linked_cell,
    const thrust::device_vector<int>& leaving_marks,
    const std::vector<PositionInfo>& position_infos) {
  // driven by RBMD_DEBUG_TRACK_GIDS
  const auto& tracked_gids = GetHaloDecisionTrackedGids();
  auto device_data = DataManager::getInstance().getDeviceData();
  auto local_box = DataManager::getInstance().getMDData()->_box.get();
  if (tracked_gids.empty() || box == nullptr || linked_cell == nullptr ||
      !device_data || local_box == nullptr) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  if (native_atoms <= 0 || total_atoms <= 0) {
    return;
  }

  const size_t native_n = static_cast<size_t>(native_atoms);
  std::vector<rbmd::Id> h_ids(native_n, 0);
  std::vector<rbmd::Real> h_px(native_n, 0);
  std::vector<rbmd::Real> h_py(native_n, 0);
  std::vector<rbmd::Real> h_pz(native_n, 0);
  std::vector<rbmd::Id> h_cell_ids(native_n, rbmd::Id(-1));
  std::vector<int> h_leaving_marks(native_n, 0);

  CHECK_RUNTIME(MEMCPY(h_ids.data(),
                       thrust::raw_pointer_cast(
                           device_data->_d_atoms_id.data()),
                       native_n * sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(MEMCPY(h_px.data(),
                       thrust::raw_pointer_cast(
                           device_data->_d_px.data()),
                       native_n * sizeof(rbmd::Real), D2H));
  CHECK_RUNTIME(MEMCPY(h_py.data(),
                       thrust::raw_pointer_cast(
                           device_data->_d_py.data()),
                       native_n * sizeof(rbmd::Real), D2H));
  CHECK_RUNTIME(MEMCPY(h_pz.data(),
                       thrust::raw_pointer_cast(
                           device_data->_d_pz.data()),
                       native_n * sizeof(rbmd::Real), D2H));
  CHECK_RUNTIME(MEMCPY(h_cell_ids.data(),
                       thrust::raw_pointer_cast(
                           linked_cell->_per_atom_cell_id.data()),
                       native_n * sizeof(rbmd::Id), D2H));
  if (leaving_marks.size() >= native_n) {
    CHECK_RUNTIME(MEMCPY(h_leaving_marks.data(),
                         thrust::raw_pointer_cast(leaving_marks.data()),
                         native_n * sizeof(int), D2H));
  }

  std::vector<Cell> h_cells(linked_cell->_cells.size());
  if (!h_cells.empty()) {
    CHECK_RUNTIME(MEMCPY(h_cells.data(),
                         thrust::raw_pointer_cast(linked_cell->_cells.data()),
                         h_cells.size() * sizeof(Cell), D2H));
  }

  EnsureCommunicationDebugLogDirectory();
  std::ofstream csv("logs/debug/tracked_gid_native_cell_state_rank" +
                        std::to_string(rank) + "_peer" +
                        std::to_string(peer_rank) + "_pid" +
                        std::to_string(static_cast<int>(::getpid())) + ".csv",
                    std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,peer_rank,gid,present_count,first_idx,first_region,"
           "px,py,pz,in_global_box,in_local_subdomain,in_peer_leaving_region,"
           "in_peer_copies_region,cell_idx,cell_is_halo,leaving_mark\n";
  }

  for (const rbmd::Id gid : tracked_gids) {
    int present_count = 0;
    rbmd::Id first_idx = -1;
    size_t first_offset = 0;
    for (size_t i = 0; i < h_ids.size(); ++i) {
      if (h_ids[i] != gid) {
        continue;
      }
      ++present_count;
      if (first_idx < 0) {
        first_idx = static_cast<rbmd::Id>(i);
        first_offset = i;
      }
    }

    rbmd::Real px = 0;
    rbmd::Real py = 0;
    rbmd::Real pz = 0;
    int in_global_box = 0;
    int in_local_subdomain = 0;
    int in_peer_leaving_region = 0;
    int in_peer_copies_region = 0;
    rbmd::Id cell_idx = -1;
    int cell_is_halo = -1;
    int leaving_mark = 0;
    if (first_idx >= 0) {
      px = h_px[first_offset];
      py = h_py[first_offset];
      pz = h_pz[first_offset];
      in_global_box =
          (box->_coord_min[0] <= px && px < box->_coord_max[0] &&
           box->_coord_min[1] <= py && py < box->_coord_max[1] &&
           box->_coord_min[2] <= pz && pz < box->_coord_max[2])
              ? 1
              : 0;
      in_local_subdomain =
          (local_box->_coord_min[0] <= px && px < local_box->_coord_max[0] &&
           local_box->_coord_min[1] <= py && py < local_box->_coord_max[1] &&
           local_box->_coord_min[2] <= pz && pz < local_box->_coord_max[2])
              ? 1
              : 0;
      for (const auto& info : position_infos) {
        if (HostIsInRegion(info._leavingLow, info._leavingHigh, px, py, pz)) {
          in_peer_leaving_region = 1;
        }
        if (HostIsInRegion(info._copiesLow, info._copiesHigh, px, py, pz)) {
          in_peer_copies_region = 1;
        }
      }
      cell_idx = h_cell_ids[first_offset];
      if (cell_idx >= 0 &&
          static_cast<size_t>(cell_idx) < h_cells.size()) {
        cell_is_halo = h_cells[static_cast<size_t>(cell_idx)]._is_halo ? 1 : 0;
      }
      leaving_mark = h_leaving_marks[first_offset] != 0 ? 1 : 0;
    }

    csv << step << "," << rank << "," << peer_rank << "," << gid << ","
        << present_count << "," << first_idx << ","
        << RegionName(first_idx, native_atoms, total_atoms) << "," << px
        << "," << py << "," << pz << "," << in_global_box << ","
        << in_local_subdomain << "," << in_peer_leaving_region << ","
        << in_peer_copies_region << "," << cell_idx << "," << cell_is_halo
        << "," << leaving_mark << "\n";
  }
}

void AppendHaloDecisionRows(
    rbmd::Id step, int rank, int peer_rank, rbmd::Id actual_start, rbmd::Id actual_end,
    rbmd::Id native_atoms, rbmd::Id ghost_atoms, rbmd::Id total_atoms,
    const std::vector<rbmd::Id>& tracked_gids,
    const std::vector<rbmd::Id>& ids_in_range,
    const std::vector<rbmd::Real>& px_in_range,
    const std::vector<rbmd::Real>& py_in_range,
    const std::vector<rbmd::Real>& pz_in_range,
    const std::vector<int>& raw_halo_marks,
    const std::vector<int>& final_halo_marks,
    const std::vector<int>& leaving_sent_mask) {
  if (tracked_gids.empty()) {
    return;
  }
  if (ids_in_range.size() != raw_halo_marks.size() ||
      raw_halo_marks.size() != final_halo_marks.size() ||
      ids_in_range.size() != px_in_range.size() ||
      ids_in_range.size() != py_in_range.size() ||
      ids_in_range.size() != pz_in_range.size()) {
    return;
  }

  std::ofstream csv("logs/debug/exchange_gid_halo_decision_rank" +
                        std::to_string(rank) + "_peer" +
                        std::to_string(peer_rank) + "_pid" +
                        std::to_string(static_cast<int>(::getpid())) + ".csv",
                    std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,rank,peer_rank,gid,actual_start,actual_end,native_atoms,"
           "ghost_atoms,total_atoms,present_count,first_idx,first_region,px,py,pz,"
           "raw_mark,leaving_mark,final_mark,excluded_by_leaving\n";
  }

  for (rbmd::Id gid : tracked_gids) {
    int present_count = 0;
    rbmd::Id first_idx = -1;
    size_t first_offset = 0;
    for (size_t i = 0; i < ids_in_range.size(); ++i) {
      if (ids_in_range[i] != gid) {
        continue;
      }
      ++present_count;
      if (first_idx < 0) {
        first_idx = actual_start + static_cast<rbmd::Id>(i);
        first_offset = i;
      }
    }

    int raw_mark = 0;
    int final_mark = 0;
    int leaving_mark = 0;
    rbmd::Real first_px = 0;
    rbmd::Real first_py = 0;
    rbmd::Real first_pz = 0;
    if (first_idx >= 0) {
      raw_mark = raw_halo_marks[first_offset] != 0 ? 1 : 0;
      final_mark = final_halo_marks[first_offset] != 0 ? 1 : 0;
      first_px = px_in_range[first_offset];
      first_py = py_in_range[first_offset];
      first_pz = pz_in_range[first_offset];
      if (first_idx >= 0 &&
          first_idx < static_cast<rbmd::Id>(leaving_sent_mask.size())) {
        leaving_mark = leaving_sent_mask[static_cast<size_t>(first_idx)] != 0 ? 1 : 0;
      }
    }

    const int excluded_by_leaving =
        (raw_mark == 1 && final_mark == 0 && leaving_mark == 1) ? 1 : 0;
    csv << step << "," << rank << "," << peer_rank << "," << gid << ","
        << actual_start << "," << actual_end << "," << native_atoms << ","
        << ghost_atoms << "," << total_atoms << "," << present_count << ","
        << first_idx << "," << RegionName(first_idx, native_atoms, total_atoms)
        << "," << first_px << "," << first_py << "," << first_pz << ","
        << raw_mark << "," << leaving_mark << "," << final_mark << ","
        << excluded_by_leaving << "\n";
  }
}

}  // namespace

// In CommunicationPartner.cpp or a similar implementation file

// --- 构造函数 (版本1: 完整参数) ---
// 使用构造函数初始化列表来正确、高效地初始化所有成员。
CommunicationPartner::CommunicationPartner(const int r, const rbmd::Real hLo[3],
                                           const rbmd::Real hHi[3],
                                           const rbmd::Real bLo[3],
                                           const rbmd::Real bHi[3],
                                           const rbmd::Real sh[3],
                                           const int offset[3],
                                           const bool enlarged[3][2])
    : _rank(r),
      // 初始化所有MPI请求为MPI_REQUEST_NULL，这是健壮MPI编程的关键
      _sendLeavingRequest(MPI_REQUEST_NULL),
      _sendHaloRequest(MPI_REQUEST_NULL),
      _recvLeavingRequest(MPI_REQUEST_NULL),
      _recvHaloRequest(MPI_REQUEST_NULL),
      // 初始化新的状态标志
      _msgSent(false),
      _isSending(false),
      _isLeavingRecvPosted(false),
      _isLeavingRecvFinalized(false),
      _isHaloRecvPosted(false),
      _isHaloRecvFinalized(false)
{
  PositionInfo p{};
  for (int d = 0; d < 3; ++d) {
    p._leavingLow[d] = hLo[d];
    p._leavingHigh[d] = hHi[d];
    p._copiesLow[d] = bLo[d];
    p._copiesHigh[d] = bHi[d];
    p._bothLow[d] = MIN(hLo[d], bLo[d]);
    p._bothHigh[d] = MAX(hHi[d], bHi[d]);
    p._shift[d] = sh[d];
    p._offset[d] = offset[d];
    p._enlarged[d][0] = enlarged[d][0];
    p._enlarged[d][1] = enlarged[d][1];
  }
  _haloInfo.push_back(p);

  _device_data = DataManager::getInstance().getDeviceData();
}

// --- 构造函数 (版本2: 只有rank) ---
CommunicationPartner::CommunicationPartner(const int r)
    : _rank(r),
      _sendLeavingRequest(MPI_REQUEST_NULL),
      _sendHaloRequest(MPI_REQUEST_NULL),
      _recvLeavingRequest(MPI_REQUEST_NULL),
      _recvHaloRequest(MPI_REQUEST_NULL),
      _msgSent(false),
      _isSending(false),
      _isLeavingRecvPosted(false),
      _isLeavingRecvFinalized(false),
      _isHaloRecvPosted(false),
      _isHaloRecvFinalized(false)
{
  PositionInfo p{};
  for (int d = 0; d < 3; ++d) {
    p._leavingLow[d] = 0.;
    p._leavingHigh[d] = 0.;
    p._copiesLow[d] = 0.;
    p._copiesHigh[d] = 0.;
    p._bothLow[d] = 0.;
    p._bothHigh[d] = 0.;
    p._shift[d] = 0.;
    p._offset[d] = 0;
    p._enlarged[d][0] = false;
    p._enlarged[d][1] = false;
  }
  _haloInfo.push_back(p);
  _device_data = DataManager::getInstance().getDeviceData();
}

// --- 构造函数 (版本3: rank和leaving区域) ---
CommunicationPartner::CommunicationPartner(const int r,
                                           const rbmd::Real leavingLo[3],
                                           const rbmd::Real leavingHigh[3])
    : _rank(r),
      _sendLeavingRequest(MPI_REQUEST_NULL),
      _sendHaloRequest(MPI_REQUEST_NULL),
      _recvLeavingRequest(MPI_REQUEST_NULL),
      _recvHaloRequest(MPI_REQUEST_NULL),
      _msgSent(false),
      _isSending(false),
      _isLeavingRecvPosted(false),
      _isLeavingRecvFinalized(false),
      _isHaloRecvPosted(false),
      _isHaloRecvFinalized(false)
{
  PositionInfo p{};
  for (int d = 0; d < 3; ++d) {
    p._leavingLow[d] = leavingLo[d];
    p._leavingHigh[d] = leavingHigh[d];
    p._copiesLow[d] = 0.;
    p._copiesHigh[d] = 0.;
    p._bothLow[d] = 0.;
    p._bothHigh[d] = 0.;
    p._shift[d] = 0.;
    p._offset[d] = 0;
    p._enlarged[d][0] = false;
    p._enlarged[d][1] = false;
  }
  _haloInfo.push_back(p);
  _device_data = DataManager::getInstance().getDeviceData();
}

// --- 拷贝构造函数 ---
// 显式地拷贝所有成员，包括新的状态标志。
CommunicationPartner::CommunicationPartner(const CommunicationPartner& o)
    : _rank(o._rank),
      _haloInfo(o._haloInfo),
      _device_data(o._device_data),
      _send_buf(o._send_buf),
      _sendLeavingRequest(o._sendLeavingRequest),
      _sendHaloRequest(o._sendHaloRequest),
      _recvLeavingRequest(o._recvLeavingRequest),
      _recvHaloRequest(o._recvHaloRequest),
      _sendLeavingStatus(o._sendLeavingStatus),
      _sendHaloStatus(o._sendHaloStatus),
      _msgSent(o._msgSent),
      _isSending(o._isSending),
      _isLeavingRecvPosted(o._isLeavingRecvPosted),
      _isLeavingRecvFinalized(o._isLeavingRecvFinalized),
      _isHaloRecvPosted(o._isHaloRecvPosted),
      _isHaloRecvFinalized(o._isHaloRecvFinalized),
      _recv_buf(o._recv_buf)
{
  // 构造函数体为空，所有工作都在初始化列表中完成。
}

// --- 赋值运算符 ---
// 正确地拷贝所有成员，并处理自我赋值的情况。
CommunicationPartner& CommunicationPartner::operator=(
    const CommunicationPartner& o) {
  // 检查自我赋值
  if (this != &o) {
    _rank = o._rank;
    _haloInfo = o._haloInfo;
    _device_data = o._device_data;
    _send_buf = o._send_buf;

    // 拷贝所有MPI句柄和状态标志
    _sendLeavingRequest = o._sendLeavingRequest;
    _sendHaloRequest = o._sendHaloRequest;
    _recvLeavingRequest = o._recvLeavingRequest;
    _recvHaloRequest = o._recvHaloRequest;
    _sendLeavingStatus = o._sendLeavingStatus;
    _sendHaloStatus = o._sendHaloStatus;
    _msgSent = o._msgSent;
    _isSending = o._isSending;
    _isLeavingRecvPosted = o._isLeavingRecvPosted;
    _isLeavingRecvFinalized = o._isLeavingRecvFinalized;
    _isHaloRecvPosted = o._isHaloRecvPosted;
    _isHaloRecvFinalized = o._isHaloRecvFinalized;
    _recv_buf = o._recv_buf;
  }
  return *this;
}

// --- 析构函数 ---
// 因为没有手动 new 任何东西，所以析构函数体为空。
CommunicationPartner::~CommunicationPartner() {
    // 无需手动 delete 任何东西。
}

void CommunicationPartner::initSend(LinkedCell* moleculeContainer,
                                    const MPI_Comm& comm, Box* box,
                                    thrust::device_vector<int>& leaving_flags,
                                    thrust::device_vector<int>& halo_flags,
                                    MessageType msgType,
                                    rbmd::Id start_atom_idx, rbmd::Id end_atom_idx,
                                    thrust::device_vector<int>* staged_leaving_marks) {
  this->CollectMoleculesInRegion(moleculeContainer, this->_haloInfo, box,
                                 leaving_flags, halo_flags, msgType,
                                 start_atom_idx, end_atom_idx,
                                 staged_leaving_marks);
  // this->_send_buf.SendUsingDerivedType(_rank,99,comm,this->_sendRequest);
  //! send leaving
  _sendLeavingRequest = MPI_REQUEST_NULL;
  _sendHaloRequest = MPI_REQUEST_NULL;
  if (this->_send_buf._leaving_atoms_double.size() > 0) {
    const auto bytes =
        this->_send_buf._leaving_atoms_double.size() * sizeof(double);
    MPI_CHECK(MPI_Isend(
        reinterpret_cast<char*>(
            thrust::raw_pointer_cast(this->_send_buf._leaving_atoms_double.data())),
        static_cast<int>(bytes), MPI_BYTE, _rank, LEAVING_TAG, comm,
        &_sendLeavingRequest));
  } else if (this->_send_buf._leaving_atoms.size() > 0) {
    MPI_CHECK(MPI_Isend(
        thrust::raw_pointer_cast(this->_send_buf._leaving_atoms.data()),
        static_cast<int>(this->_send_buf._leaving_atoms.size()), MPI_BYTE, _rank,
        LEAVING_TAG, comm, &_sendLeavingRequest));
  }
  if (this->_send_buf._ghost_atoms.size() > 0) {
    MPI_CHECK(
    MPI_Isend(thrust::raw_pointer_cast(this->_send_buf._ghost_atoms.data()),
      this->_send_buf._ghost_atoms.size(),
      MPI_BYTE, _rank,
      HALO_TAG, comm,  &_sendHaloRequest));
  }
  _msgSent = false;
  _isSending = true;
}

void CommunicationPartner::PrepareMessage(
    LinkedCell* moleculeContainer, Box* box,
    thrust::device_vector<int>& leaving_flags,
    thrust::device_vector<int>& halo_flags, MessageType msgType,
    const MPI_Comm& comm, int current_rank, int total_rank,
    rbmd::Id start_atom_idx, rbmd::Id end_atom_idx,
    thrust::device_vector<int>* staged_leaving_marks) {

  this->CollectMoleculesInRegion(moleculeContainer, this->_haloInfo, box,
                                 leaving_flags, halo_flags, msgType,
                                 start_atom_idx, end_atom_idx,
                                 staged_leaving_marks);
  
  // // 如果提供了 MPI 参数，则输出 HaloInfo 到文件
  // if (current_rank >= 0 && total_rank > 0) {
  //   outputHaloInfoToFile(current_rank, total_rank, comm);
  // }
}

int CommunicationPartner::GetSendCount(MessageType msgType) const {
  switch (msgType) {
    case LEAVING_ONLY:
    case LEAVING_AND_HALO_COPIES:
      // T034: leaving 使用 LAMMPS layout double buffer（按原子计数）
      if (!_send_buf._leaving_atoms_double.empty() && _send_buf._size_exchange > 0) {
        return static_cast<int>(_send_buf._leaving_atoms_double.size() /
                                static_cast<size_t>(_send_buf._size_exchange));
      }
      // 兼容旧模式：按字节布局估算数量
      if (_send_buf._leaving_size == 0) return 0;
      return static_cast<int>(_send_buf._leaving_atoms.size() /
                              _send_buf._leaving_size);
    case HALO_COPIES:
      if (_send_buf._ghost_size == 0) return 0;
      return static_cast<int>(_send_buf._ghost_atoms.size() /
                              _send_buf._ghost_size);
    case FORCES:
      return 0;
    default:
      return 0;
  }
}

size_t CommunicationPartner::GetSendBytes(MessageType msgType) const {
  switch (msgType) {
    case LEAVING_ONLY:
    case LEAVING_AND_HALO_COPIES:
      // T034: leaving 走 double buffer（单位：byte）
      if (!_send_buf._leaving_atoms_double.empty()) {
        return _send_buf._leaving_atoms_double.size() * sizeof(double);
      }
      return _send_buf._leaving_atoms.size();
    case HALO_COPIES:
      return _send_buf._ghost_atoms.size();
    case FORCES:
      return 0;
    default:
      return 0;
  }
}

char* CommunicationPartner::GetSendBufferPtr(MessageType msgType) {
  switch (msgType) {
    case LEAVING_ONLY:
    case LEAVING_AND_HALO_COPIES:
      // T034: leaving 走 double buffer
      if (!_send_buf._leaving_atoms_double.empty()) {
        auto* ptr =
            thrust::raw_pointer_cast(_send_buf._leaving_atoms_double.data());
        return reinterpret_cast<char*>(ptr);
      }
      return _send_buf._leaving_atoms.empty()
                 ? nullptr
                 : thrust::raw_pointer_cast(_send_buf._leaving_atoms.data());
    case HALO_COPIES:
      return _send_buf._ghost_atoms.empty()
                 ? nullptr
                 : thrust::raw_pointer_cast(_send_buf._ghost_atoms.data());
    case FORCES:
      return nullptr;
    default:
      return nullptr;
  }
}

void CommunicationPartner::EnsureRecvBuffer(MessageType msgType,
                                            size_t recvBytes) {
  const auto atom_style =
      DataManager::getInstance().getConfigData()->Get<std::string>(
          "atom_style", "init_configuration", "read_data");
  const bool use_topology_exchange = UsesTopologyExchangeLayout(atom_style);
  switch (msgType) {
    case LEAVING_ONLY:
    case LEAVING_AND_HALO_COPIES:
      if (recvBytes == 0) {
        _recv_buf._leaving_atoms_double.clear();
        _recv_buf._leaving_atoms.clear();
        break;
      }
      if (use_topology_exchange) {
        if (recvBytes % sizeof(double) != 0) {
          throw std::runtime_error(
              std::string("LEAVING recvBytes not aligned to double: ") +
              std::to_string(recvBytes));
        }
        _recv_buf._leaving_atoms_double.resize(recvBytes / sizeof(double));
        _recv_buf._leaving_atoms.clear();
      } else {
        _recv_buf._leaving_atoms.clear();
        _recv_buf.ResizeLeavingByByteSize(recvBytes);
        _recv_buf._leaving_atoms_double.clear();
      }
      break;
    case HALO_COPIES:
      _recv_buf.ResizeGhostByByteSize(recvBytes);
      break;
    case FORCES:
      break;
    default:
      break;
  }
}

char* CommunicationPartner::GetRecvBufferPtr(MessageType msgType) {
  switch (msgType) {
    case LEAVING_ONLY:
    case LEAVING_AND_HALO_COPIES:
      // T036: leaving 走 double buffer
      if (!_recv_buf._leaving_atoms_double.empty()) {
        auto* ptr =
            thrust::raw_pointer_cast(_recv_buf._leaving_atoms_double.data());
        return reinterpret_cast<char*>(ptr);
      }
      return _recv_buf._leaving_atoms.empty()
                 ? nullptr
                 : thrust::raw_pointer_cast(_recv_buf._leaving_atoms.data());
    case HALO_COPIES:
      return _recv_buf._ghost_atoms.empty()
                 ? nullptr
                 : thrust::raw_pointer_cast(_recv_buf._ghost_atoms.data());
    case FORCES:
      return nullptr;
    default:
      return nullptr;
  }
}

void CommunicationPartner::ClearSendBuffer() { _send_buf.Clear(); }

void CommunicationPartner::PrepareShakeForwardMessage(
    const thrust::host_vector<rbmd::Id>& atom_ids,
    const thrust::host_vector<rbmd::Real>& values) {
  const size_t bytes =
      atom_ids.size() * sizeof(rbmd::Id) + values.size() * sizeof(rbmd::Real);
  _send_buf.ResizeShakeForwardByByteSize(bytes);
}

void CommunicationPartner::PrepareShakeReverseMessage(
    const thrust::host_vector<rbmd::Id>& atom_ids,
    const thrust::host_vector<rbmd::Real>& values) {
  const size_t bytes =
      atom_ids.size() * sizeof(rbmd::Id) + values.size() * sizeof(rbmd::Real);
  _send_buf.ResizeShakeReverseByByteSize(bytes);
}

void CommunicationPartner::ProcessShakeForwardData() {
  if (_recv_buf._shake_forward_atoms.empty()) {
    return;
  }
}

void CommunicationPartner::ProcessShakeReverseData() {
  if (_recv_buf._shake_reverse_atoms.empty()) {
    return;
  }
}

bool CommunicationPartner::testSend() {
  if (not _msgSent) {
    int flag_leaving = 1; // 默认完成
    int flag_halo = 1;    // 默认完成

    // 只测试非空的请求
    if (_sendLeavingRequest != MPI_REQUEST_NULL) {
      flag_leaving = 0; // 先假设未完成
      MPI_CHECK(MPI_Test(&_sendLeavingRequest, &flag_leaving, &_sendLeavingStatus));
    }

    if (_sendHaloRequest != MPI_REQUEST_NULL) {
      flag_halo = 0; // 先假设未完成
      MPI_CHECK(MPI_Test(&_sendHaloRequest, &flag_halo, &_sendHaloStatus));
    }

    if (flag_leaving == 1 && flag_halo == 1) {
      _msgSent = true;
      _isSending = false;
      _send_buf.Clear();

      // 完成后，将请求句柄重置为 NULL 是一个好习惯
      _sendLeavingRequest = MPI_REQUEST_NULL;
      _sendHaloRequest = MPI_REQUEST_NULL;
    }
  }
  return _msgSent;
}


void CommunicationPartner::ResetReceive() {
  // 重置 Leaving 消息的状态
  _isLeavingRecvPosted = false;
  _isLeavingRecvFinalized = false;
  _recvLeavingRequest = MPI_REQUEST_NULL;
  // 重置 Halo 消息的状态
  _isHaloRecvPosted = false;
  _isHaloRecvFinalized = false;
  _recvHaloRequest = MPI_REQUEST_NULL;
  _recv_buf.Clear();

}

void CommunicationPartner::iprobeAndPostRecv(MPI_Comm comm, int tag) {
  // 根据tag决定要操作的状态变量
  bool& isRecvPosted = (tag == LEAVING_TAG) ? _isLeavingRecvPosted : _isHaloRecvPosted;

  // 如果已经为这个tag发出了Irecv，就直接返回，避免重复操作
  if (isRecvPosted) {
    return;
  }

  int msg_available = 0;
  MPI_Status status;
  MPI_CHECK(MPI_Iprobe(_rank, tag, comm, &msg_available, &status));

  if (msg_available) {
    int num_bytes;
    MPI_CHECK(MPI_Get_count(&status, MPI_BYTE, &num_bytes));
    const auto atom_style =
        DataManager::getInstance().getConfigData()->Get<std::string>(
            "atom_style", "init_configuration", "read_data");
    const bool use_topology_exchange = UsesTopologyExchangeLayout(atom_style);

    // 始终为匹配消息张贴 Irecv，即便消息大小为 0
    if (tag == LEAVING_TAG) {
        if (use_topology_exchange) {
          if (num_bytes > 0) {
            if (num_bytes % static_cast<int>(sizeof(double)) != 0) {
              throw std::runtime_error(
                  std::string("LEAVING message bytes not aligned to double: ") +
                  std::to_string(num_bytes));
            }
            _recv_buf._leaving_atoms_double.resize(
                static_cast<size_t>(num_bytes) / sizeof(double));
            _recv_buf._leaving_atoms.clear();
          } else {
            _recv_buf._leaving_atoms_double.clear();
            _recv_buf._leaving_atoms.clear();
          }
          char* recv_ptr =
              num_bytes > 0
                  ? reinterpret_cast<char*>(thrust::raw_pointer_cast(
                        _recv_buf._leaving_atoms_double.data()))
                  : nullptr;
          MPI_CHECK(MPI_Irecv(recv_ptr, num_bytes, MPI_BYTE, _rank, LEAVING_TAG,
                              comm, &_recvLeavingRequest));
        } else {
          if (num_bytes > 0) {
            _recv_buf.ResizeLeavingByByteSize(static_cast<size_t>(num_bytes));
          } else {
            _recv_buf._leaving_atoms.clear();
          }
          _recv_buf._leaving_atoms_double.clear();
          char* recv_ptr = num_bytes > 0
                               ? thrust::raw_pointer_cast(
                                     _recv_buf._leaving_atoms.data())
                               : nullptr;
          MPI_CHECK(MPI_Irecv(recv_ptr, num_bytes, MPI_BYTE, _rank, LEAVING_TAG,
                              comm, &_recvLeavingRequest));
        }
    } else {  // HALO_TAG
        _recv_buf.ResizeGhostByByteSize(num_bytes);
        char* recv_ptr = num_bytes > 0 ?
            thrust::raw_pointer_cast(_recv_buf._ghost_atoms.data()) : nullptr;
        MPI_CHECK(MPI_Irecv(recv_ptr, num_bytes, MPI_BYTE,
                            _rank, HALO_TAG, comm, &_recvHaloRequest));
    }
    // 标记已经为此 tag 启动了 Irecv（即使是 0 字节消息）
    isRecvPosted = true;
  }
}




// 将接收到的返回容器中

bool CommunicationPartner::testRecv(
    LinkedCell *linked_cell, bool force) {
  // 检查 Leaving 粒子的接收
  if (_isLeavingRecvPosted && !_isLeavingRecvFinalized) {
    int flag = 0;
    MPI_CHECK(MPI_Test(&_recvLeavingRequest, &flag, MPI_STATUS_IGNORE));
    if (flag) {
      _isLeavingRecvFinalized = true;
    }
  }

  // 检查 Halo 粒子的接收
  if (_isHaloRecvPosted && !_isHaloRecvFinalized) {
    int flag = 0;
    MPI_CHECK(MPI_Test(&_recvHaloRequest, &flag, MPI_STATUS_IGNORE));
    if (flag) {
      _isHaloRecvFinalized = true;
    }
  }

  // 如果一个Irecv从未被发出(isRecvPosted=false)，那么它也被认为是“完成”的。
  // 只有当Irecv被发出(isRecvPosted=true)但尚未处理完(isRecvFinalized=false)时，才算未完成。
  bool leavingDone = !_isLeavingRecvPosted || _isLeavingRecvFinalized;
  bool haloDone = !_isHaloRecvPosted || _isHaloRecvFinalized;

  return leavingDone && haloDone;
}

void CommunicationPartner::add(CommunicationPartner partner) {
  //RBMD_ASSERT(partner._rank == _rank);
  _haloInfo.push_back(partner._haloInfo[0]);
}

//! TODO 好像不用
size_t CommunicationPartner::getDynamicSize() {
  return _haloInfo.capacity() * sizeof(PositionInfo);
  // _sendBuf.getDynamicSize() + _recvBuf.getDynamicSize() +
  // _haloInfo.capacity() * sizeof(PositionInfo) TODO
}

void CommunicationPartner::print(std::ostream& stream) const {
  stream << "Partner rank: " << _rank << std::endl;
  stream << "Halo regions: " << std::endl;
  for (auto& region : _haloInfo) {
    stream << " Region:" << std::endl;
    stream << "  both: " << " [" << region._bothLow[0] << ", "
        << region._bothHigh[0] << ") x " << " [" << region._bothLow[1]
        << ", " << region._bothHigh[1] << ") x " << " ["
        << region._bothLow[2] << ", " << region._bothHigh[2] << ")"
        << std::endl;
    stream << "  leaving: " << " [" << region._leavingLow[0] << ", "
        << region._leavingHigh[0] << ") x " << " [" << region._leavingLow[1]
        << ", " << region._leavingHigh[1] << ") x " << " ["
        << region._leavingLow[2] << ", " << region._leavingHigh[2] << ")"
        << std::endl;
    stream << "  copies: " << " [" << region._copiesLow[0] << ", "
        << region._copiesHigh[0] << ") x " << " [" << region._copiesLow[1]
        << ", " << region._copiesHigh[1] << ") x " << " ["
        << region._copiesLow[2] << ", " << region._copiesHigh[2] << ")"
        << std::endl;
    stream << "  offset: (" << region._offset[0] << ", " << region._offset[1]
        << ", " << region._offset[2] << ")" << std::endl;
    stream << "  shift:	(" << region._shift[0] << ", " << region._shift[1]
        << ", " << region._shift[2] << ")" << std::endl;
  }
}

void CommunicationPartner::ProcessLeavingData(LinkedCell* linked_cell) {
  const auto atom_style =
      DataManager::getInstance().getConfigData()->Get<std::string>(
          "atom_style", "init_configuration", "read_data");
  if (atom_style != "full") {
    const auto byte_size = _recv_buf._leaving_atoms.size();
    const unsigned int recv_leaving_atoms_num =
        byte_size / _recv_buf._leaving_size;
    if (recv_leaving_atoms_num == 0) {
      return;
    }

    char* p_recv_buffer = thrust::raw_pointer_cast(_recv_buf._leaving_atoms.data());
    const size_t id_size = sizeof(rbmd::Id);
    const size_t real_size = sizeof(rbmd::Real);
    auto device_data = DataManager::getInstance().getDeviceData();
    const bool has_charge = atom_style == "charge" || atom_style == "full";
    const bool has_molecule = atom_style == "full";

    const rbmd::Id recv_count = static_cast<rbmd::Id>(recv_leaving_atoms_num);
    const rbmd::Id old_native = linked_cell->_native_atoms_num;
    const rbmd::Id new_native_total = old_native + recv_count;
    const rbmd::Id target_total = new_native_total;

    linked_cell->EnsureCapacity(target_total);
    linked_cell->ResizeNativeDependentVectors(new_native_total, target_total);

    size_t current_offset = 0;
    const rbmd::Id shared_insert = old_native;
    const rbmd::Id velocity_insert = old_native;
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Id*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Id> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Id> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_atoms_id.begin() + shared_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * id_size;
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Id*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Id> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Id> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_atoms_type.begin() + shared_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * id_size;
    if (has_molecule) {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Id*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Id> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Id> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter =
          device_data->_d_molecular_id.begin() + shared_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
      current_offset += recv_leaving_atoms_num * id_size;
    }
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Real*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Real> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_px.begin() + shared_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * real_size;
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Real*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Real> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_py.begin() + shared_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * real_size;
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Real*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Real> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_pz.begin() + shared_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * real_size;
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Real*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Real> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_vx.begin() + velocity_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * real_size;
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Real*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Real> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_vy.begin() + velocity_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * real_size;
    {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Real*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Real> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_vz.begin() + velocity_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }
    current_offset += recv_leaving_atoms_num * real_size;
    if (has_charge) {
      auto source_raw_ptr =
          reinterpret_cast<rbmd::Real*>(p_recv_buffer + current_offset);
      thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
      thrust::device_ptr<rbmd::Real> source_end_iter =
          source_begin_iter + recv_leaving_atoms_num;
      auto dest_begin_iter = device_data->_d_charge.begin() + shared_insert;
      thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
    }

    thrust::fill(linked_cell->_per_atom_cell_id.begin() + shared_insert,
                 linked_cell->_per_atom_cell_id.begin() + shared_insert +
                     recv_count,
                 0);
    return;
  }

  const size_t total_doubles = _recv_buf._leaving_atoms_double.size();
  const unsigned int recv_leaving_atoms_num =
      (_recv_buf._size_exchange > 0)
          ? static_cast<unsigned int>(total_doubles / _recv_buf._size_exchange)
          : 0;
  
  if (recv_leaving_atoms_num == 0) {
    return;
  }

  auto device_data = DataManager::getInstance().getDeviceData();

  const rbmd::Id recv_count =
      static_cast<rbmd::Id>(recv_leaving_atoms_num);
  const rbmd::Id old_native = linked_cell->_native_atoms_num;
  // 注：ClearDataHalo() 已被调用，_ghost_atoms_num 为 0
  const rbmd::Id new_native_total = old_native + recv_count;
  const rbmd::Id target_total = new_native_total;  // ghost_atoms = 0

#ifndef NDEBUG
  // Topology 数组是按 DeviceData::nmax 预分配的（与 LinkedCell 的容量概念不同）。
  // 如果迁移后 native 超过 nmax，会导致 unpack 写入 per-atom topology 数组 OOB。
  if (device_data && device_data->nmax > 0 &&
      new_native_total > device_data->nmax) {
    std::cerr << "[ProcessLeavingData] ERROR: native overflow vs DeviceData::nmax "
              << "old_native=" << old_native << " recv_count=" << recv_count
              << " new_native_total=" << new_native_total
              << " nmax=" << device_data->nmax << std::endl;
    assert(false);
  }
#endif

  // 使用简化的容量管理：直接确保总容量足够
  linked_cell->EnsureCapacity(target_total);
  linked_cell->ResizeNativeDependentVectors(new_native_total, target_total);
  // rbsog 的 ResizeNativeDependentVectors() 已经同步更新
  // _native/_ghost/_total 计数；此处不能再做一次 native 计数累加，
  // 否则会重复累加。

  // T036: 调用 UnpackExchangeTopologyOp (对齐 LAMMPS)
  // 创建 device-side nlocal 计数器（初始化为 old_native）
  thrust::device_vector<int> d_nlocal(1);
  d_nlocal[0] = CheckedIntCount(old_native, "topology unpack old_native");

  // 填充 TopologyUnpackParams
  op::TopologyUnpackParams unpack_params{};
  unpack_params.nrecv =
      CheckedIntCount(recv_count, "topology unpack recv_count");
  unpack_params.buf = thrust::raw_pointer_cast(
      _recv_buf._leaving_atoms_double.data());
  unpack_params.size_exchange = _recv_buf._size_exchange;
  unpack_params.nlocal_ptr = thrust::raw_pointer_cast(d_nlocal.data());
  const int unpack_nmax =
      (device_data->nmax > 0)
          ? CheckedIntCount(device_data->nmax, "topology unpack nmax")
          : CheckedIntCount(
                static_cast<rbmd::Id>(device_data->_d_atoms_id.size()),
                "topology unpack atom capacity");
  if (unpack_nmax <= 0) {
    throw std::runtime_error(
        "Invalid topology unpack nmax (<=0), cannot unpack leaving atoms.");
  }
  unpack_params.nmax = unpack_nmax;

  // 基础属性
  unpack_params.d_px = thrust::raw_pointer_cast(device_data->_d_px.data());
  unpack_params.d_py = thrust::raw_pointer_cast(device_data->_d_py.data());
  unpack_params.d_pz = thrust::raw_pointer_cast(device_data->_d_pz.data());
  unpack_params.d_vx = thrust::raw_pointer_cast(device_data->_d_vx.data());
  unpack_params.d_vy = thrust::raw_pointer_cast(device_data->_d_vy.data());
  unpack_params.d_vz = thrust::raw_pointer_cast(device_data->_d_vz.data());
  unpack_params.d_id =
      thrust::raw_pointer_cast(device_data->_d_atoms_id.data());
  unpack_params.d_type =
      thrust::raw_pointer_cast(device_data->_d_atoms_type.data());
  unpack_params.d_molecule =
      thrust::raw_pointer_cast(device_data->_d_molecular_id.data());
  unpack_params.d_charge =
      thrust::raw_pointer_cast(device_data->_d_charge.data());

  // Bond 数据
  unpack_params.d_num_bond =
      thrust::raw_pointer_cast(device_data->d_num_bond.data());
  unpack_params.d_bond_type =
      thrust::raw_pointer_cast(device_data->d_bond_type.data());
  unpack_params.d_bond_atom =
      thrust::raw_pointer_cast(device_data->d_bond_atom.data());
  unpack_params.bond_per_atom = device_data->bond_per_atom;

  // Angle 数据
  unpack_params.d_num_angle =
      thrust::raw_pointer_cast(device_data->d_num_angle.data());
  unpack_params.d_angle_type =
      thrust::raw_pointer_cast(device_data->d_angle_type.data());
  unpack_params.d_angle_atom1 =
      thrust::raw_pointer_cast(device_data->d_angle_atom1.data());
  unpack_params.d_angle_atom2 =
      thrust::raw_pointer_cast(device_data->d_angle_atom2.data());
  unpack_params.d_angle_atom3 =
      thrust::raw_pointer_cast(device_data->d_angle_atom3.data());
  unpack_params.angle_per_atom = device_data->angle_per_atom;

  // Dihedral 数据
  unpack_params.d_num_dihedral =
      thrust::raw_pointer_cast(device_data->d_num_dihedral.data());
  unpack_params.d_dihedral_type =
      thrust::raw_pointer_cast(device_data->d_dihedral_type.data());
  unpack_params.d_dihedral_atom1 =
      thrust::raw_pointer_cast(device_data->d_dihedral_atom1.data());
  unpack_params.d_dihedral_atom2 =
      thrust::raw_pointer_cast(device_data->d_dihedral_atom2.data());
  unpack_params.d_dihedral_atom3 =
      thrust::raw_pointer_cast(device_data->d_dihedral_atom3.data());
  unpack_params.d_dihedral_atom4 =
      thrust::raw_pointer_cast(device_data->d_dihedral_atom4.data());
  unpack_params.dihedral_per_atom = device_data->dihedral_per_atom;

  // Improper 数据
  unpack_params.d_num_improper =
      thrust::raw_pointer_cast(device_data->d_num_improper.data());
  unpack_params.d_improper_type =
      thrust::raw_pointer_cast(device_data->d_improper_type.data());
  unpack_params.d_improper_atom1 =
      thrust::raw_pointer_cast(device_data->d_improper_atom1.data());
  unpack_params.d_improper_atom2 =
      thrust::raw_pointer_cast(device_data->d_improper_atom2.data());
  unpack_params.d_improper_atom3 =
      thrust::raw_pointer_cast(device_data->d_improper_atom3.data());
  unpack_params.d_improper_atom4 =
      thrust::raw_pointer_cast(device_data->d_improper_atom4.data());
  unpack_params.improper_per_atom = device_data->improper_per_atom;

  // Special bond 数据
  unpack_params.d_nspecial =
      thrust::raw_pointer_cast(device_data->d_nspecial.data());
  unpack_params.d_special =
      thrust::raw_pointer_cast(device_data->d_special.data());
  unpack_params.maxspecial = device_data->maxspecial;

  // 调用 unpack op (T030)
  op::UnpackExchangeTopologyOp<device::DEVICE_GPU> unpack_op;
  unpack_op(unpack_params);

  // ========== T037: Debug 模式验证接收侧 unpack（仅 NDEBUG 未定义时启用） ==========
#ifndef NDEBUG
  // 验证 nlocal 更新是否正确（避免生产路径引入隐式 D2H）
  int final_nlocal = d_nlocal[0];
  if (final_nlocal !=
      CheckedIntCount(new_native_total, "topology unpack new_native_total")) {
    std::cerr << "[ProcessLeavingData] ERROR: nlocal mismatch after unpack! "
              << "Expected: " << new_native_total
              << ", Got: " << final_nlocal << std::endl;
  }

#endif  // NDEBUG
  // ========== T037 验证结束 ==========

  // 初始化新原子的 cell_id
  thrust::fill(linked_cell->_per_atom_cell_id.begin() + old_native,
               linked_cell->_per_atom_cell_id.begin() + new_native_total,
               0);
}

void CommunicationPartner::ProcessGhostData(LinkedCell* linked_cell) {
  const auto byte_size = _recv_buf._ghost_atoms.size();
  const unsigned int recv_ghost_atoms_num =
      byte_size / _recv_buf._ghost_size;
  if (recv_ghost_atoms_num == 0) {
    return;
  }
  char* p_recv_buffer = thrust::raw_pointer_cast(
      _recv_buf._ghost_atoms.data());
  const size_t id_size = sizeof(rbmd::Id);
  const size_t real_size = sizeof(rbmd::Real);
  auto device_data = DataManager::getInstance().getDeviceData();
  const auto atom_style =
      DataManager::getInstance().getConfigData()->Get<std::string>(
          "atom_style", "init_configuration", "read_data");
  const bool has_charge =
      atom_style == "charge" || atom_style == "full";

  rbmd::Id recv_count = static_cast<rbmd::Id>(recv_ghost_atoms_num);
  const rbmd::Id old_total = linked_cell->_total_atoms_num;
  const rbmd::Id old_native = linked_cell->_native_atoms_num;

  if (DedupGhostAtomsEnabled()) {
    size_t dedup_offset = 0;
    auto recv_ids = CopyGhostColumnToHost<rbmd::Id>(
        p_recv_buffer, dedup_offset, recv_ghost_atoms_num);

    std::unordered_set<rbmd::Id> seen_ids;
    seen_ids.reserve(static_cast<size_t>(old_total) +
                     static_cast<size_t>(recv_ghost_atoms_num));
    if (old_total > 0) {
      thrust::host_vector<rbmd::Id> existing_ids(
          static_cast<size_t>(old_total));
      thrust::copy(device_data->_d_atoms_id.begin(),
                   device_data->_d_atoms_id.begin() + old_total,
                   existing_ids.begin());
      for (const auto id : existing_ids) {
        seen_ids.insert(id);
      }
    }

    std::vector<unsigned int> keep_indices;
    keep_indices.reserve(recv_ghost_atoms_num);
    for (unsigned int i = 0; i < recv_ghost_atoms_num; ++i) {
      if (seen_ids.insert(recv_ids[i]).second) {
        keep_indices.push_back(i);
      }
    }

    if (keep_indices.size() != recv_ghost_atoms_num) {
      if (keep_indices.empty()) {
        return;
      }

      dedup_offset += recv_ghost_atoms_num * id_size;
      auto recv_types = CopyGhostColumnToHost<rbmd::Id>(
          p_recv_buffer, dedup_offset, recv_ghost_atoms_num);
      dedup_offset += recv_ghost_atoms_num * id_size;
      auto recv_px = CopyGhostColumnToHost<rbmd::Real>(
          p_recv_buffer, dedup_offset, recv_ghost_atoms_num);
      dedup_offset += recv_ghost_atoms_num * real_size;
      auto recv_py = CopyGhostColumnToHost<rbmd::Real>(
          p_recv_buffer, dedup_offset, recv_ghost_atoms_num);
      dedup_offset += recv_ghost_atoms_num * real_size;
      auto recv_pz = CopyGhostColumnToHost<rbmd::Real>(
          p_recv_buffer, dedup_offset, recv_ghost_atoms_num);
      dedup_offset += recv_ghost_atoms_num * real_size;
      thrust::host_vector<rbmd::Real> recv_charge;
      if (has_charge) {
        recv_charge = CopyGhostColumnToHost<rbmd::Real>(
            p_recv_buffer, dedup_offset, recv_ghost_atoms_num);
      }

      auto kept_ids = GatherHostByIndex(recv_ids, keep_indices);
      auto kept_types = GatherHostByIndex(recv_types, keep_indices);
      auto kept_px = GatherHostByIndex(recv_px, keep_indices);
      auto kept_py = GatherHostByIndex(recv_py, keep_indices);
      auto kept_pz = GatherHostByIndex(recv_pz, keep_indices);
      thrust::host_vector<rbmd::Real> kept_charge;
      if (has_charge) {
        kept_charge = GatherHostByIndex(recv_charge, keep_indices);
      }

      recv_count = static_cast<rbmd::Id>(keep_indices.size());
      const rbmd::Id new_ghost = linked_cell->_ghost_atoms_num + recv_count;
      const rbmd::Id target_total = old_native + new_ghost;
      const rbmd::Id insertion_offset = old_total;

      linked_cell->EnsureCapacity(target_total);
      linked_cell->ResizeNativeDependentVectors(old_native, target_total);

      thrust::copy(kept_ids.begin(), kept_ids.end(),
                   device_data->_d_atoms_id.begin() + insertion_offset);
      thrust::copy(kept_types.begin(), kept_types.end(),
                   device_data->_d_atoms_type.begin() + insertion_offset);
      thrust::copy(kept_px.begin(), kept_px.end(),
                   device_data->_d_px.begin() + insertion_offset);
      thrust::copy(kept_py.begin(), kept_py.end(),
                   device_data->_d_py.begin() + insertion_offset);
      thrust::copy(kept_pz.begin(), kept_pz.end(),
                   device_data->_d_pz.begin() + insertion_offset);
      if (has_charge) {
        thrust::copy(kept_charge.begin(), kept_charge.end(),
                     device_data->_d_charge.begin() + insertion_offset);
      }

      thrust::fill(linked_cell->_per_atom_cell_id.begin() + insertion_offset,
                   linked_cell->_per_atom_cell_id.begin() + insertion_offset +
                       recv_count,
                   0);
      return;
    }
  }

  const rbmd::Id new_ghost = linked_cell->_ghost_atoms_num + recv_count;
  const rbmd::Id target_total = old_native + new_ghost;

  // 使用简化的容量管理：直接确保总容量足够
  linked_cell->EnsureCapacity(target_total);
  linked_cell->ResizeNativeDependentVectors(old_native, target_total);
  // rbsog 的 ResizeNativeDependentVectors() 已经同步更新
  // _native/_ghost/_total 计数；此处不能再做一次 ghost 计数累加，
  // 否则会重复累加。

  size_t current_offset = 0;
  const rbmd::Id insertion_offset = old_total;
  {
    auto source_raw_ptr = reinterpret_cast<rbmd::Id*>(
        p_recv_buffer + current_offset);
    thrust::device_ptr<rbmd::Id> source_begin_iter(source_raw_ptr);
    thrust::device_ptr<rbmd::Id> source_end_iter =
        source_begin_iter + recv_ghost_atoms_num;
    auto dest_begin_iter =
        device_data->_d_atoms_id.begin() + insertion_offset;
    thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
  }
  current_offset += recv_ghost_atoms_num * id_size;
  {
    auto source_raw_ptr = reinterpret_cast<rbmd::Id*>(
        p_recv_buffer + current_offset);
    thrust::device_ptr<rbmd::Id> source_begin_iter(source_raw_ptr);
    thrust::device_ptr<rbmd::Id> source_end_iter =
        source_begin_iter + recv_ghost_atoms_num;
    auto dest_begin_iter =
        device_data->_d_atoms_type.begin() + insertion_offset;
    thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
  }
  current_offset += recv_ghost_atoms_num * id_size;
  {
    auto source_raw_ptr = reinterpret_cast<rbmd::Real*>(
        p_recv_buffer + current_offset);
    thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
    thrust::device_ptr<rbmd::Real> source_end_iter =
        source_begin_iter + recv_ghost_atoms_num;
    auto dest_begin_iter = device_data->_d_px.begin() + insertion_offset;
    thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
  }
  current_offset += recv_ghost_atoms_num * real_size;
  {
    auto source_raw_ptr = reinterpret_cast<rbmd::Real*>(
        p_recv_buffer + current_offset);
    thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
    thrust::device_ptr<rbmd::Real> source_end_iter =
        source_begin_iter + recv_ghost_atoms_num;
    auto dest_begin_iter = device_data->_d_py.begin() + insertion_offset;
    thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
  }
  current_offset += recv_ghost_atoms_num * real_size;
  {
    auto source_raw_ptr = reinterpret_cast<rbmd::Real*>(
        p_recv_buffer + current_offset);
    thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
    thrust::device_ptr<rbmd::Real> source_end_iter =
        source_begin_iter + recv_ghost_atoms_num;
    auto dest_begin_iter = device_data->_d_pz.begin() + insertion_offset;
    thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
  }
  current_offset += recv_ghost_atoms_num * real_size;
  if (has_charge) {
    auto source_raw_ptr = reinterpret_cast<rbmd::Real*>(
        p_recv_buffer + current_offset);
    thrust::device_ptr<rbmd::Real> source_begin_iter(source_raw_ptr);
    thrust::device_ptr<rbmd::Real> source_end_iter =
        source_begin_iter + recv_ghost_atoms_num;
    auto dest_begin_iter = device_data->_d_charge.begin() + insertion_offset;
    thrust::copy(source_begin_iter, source_end_iter, dest_begin_iter);
  }

  thrust::fill(linked_cell->_per_atom_cell_id.begin() + insertion_offset,
               linked_cell->_per_atom_cell_id.begin() + insertion_offset +
                   recv_count,
               0);
}

// 收集需要发送给邻居进程的原子（leaving或halo）
// 
// 根据文档3.2.3节的三阶段通信（Staged Messaging）方案：
// 1. 在每个维度依次进行通信（x维 -> y维 -> z维）
// 2. 后续阶段会处理前面阶段接收到的halo原子，使其能继续传播
// 
// 参数 end_atom_idx 的含义：
// - 对于 LEAVING_ONLY：传0（使用默认值），函数内部使用 native_num
// - 对于 HALO_COPIES：传当前 _total_atoms_num，表示要处理的原子总数
//   * 第1阶段（x维）：end_atom_idx = native_num（只处理native原子）
//   * 第2阶段（y维）：end_atom_idx = native_num + x维halo数
//   * 第3阶段（z维）：end_atom_idx = native_num + x维halo + y维halo数
//
// 这样实现了文档描述的"在每个阶段中将前一阶段接收的数据加到后续消息中"
void CommunicationPartner::CollectMoleculesInRegion(
    LinkedCell* linked_cell,
    std::vector<PositionInfo>& position_infos, Box* box,
    thrust::device_vector<int>& leaving_flags,
    thrust::device_vector<int>& halo_flags, MessageType msgType,
    rbmd::Id start_atom_idx, rbmd::Id end_atom_idx,
    thrust::device_vector<int>* staged_leaving_marks) {
  this->_send_buf.Clear();

  auto atom_style =
      DataManager::getInstance().getConfigData()->Get<std::string>(
          "atom_style", "init_configuration", "read_data");
  thrust::device_vector<int>* deferred_mask = staged_leaving_marks;
  const bool track_deferred_deletion =
      deferred_mask != nullptr &&
      (msgType == LEAVING_ONLY || msgType == LEAVING_AND_HALO_COPIES);

  // ========== 阶段 1: 处理 Leaving 粒子 ==========
  if (msgType == LEAVING_ONLY || msgType == LEAVING_AND_HALO_COPIES) {
    op::MarkLeavingAtomsInRegionOp<device::DEVICE_GPU>
        mark_leaving_atoms_in_region_op;

    struct CachedRegionData {
      thrust::device_vector<int> indices;
      int count;
      size_t info_idx;
    };
    std::vector<CachedRegionData> cached_regions;
    cached_regions.reserve(position_infos.size());
    size_t total_leaving_atoms = 0;
    // 仅在本次 Collect 调用内使用：避免同一 partner 的多个 region 重复打包。
    thrust::device_vector<int> local_marked_in_call;

    // Pass 1: Identify atoms, filter duplicates, and cache indices
    for (size_t info_idx = 0; info_idx < position_infos.size(); ++info_idx) {
      const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
      if (native_atoms == 0) {
        break;
      }
      const size_t native_atom_size = static_cast<size_t>(native_atoms);
      leaving_flags.resize(native_atom_size);
      thrust::fill(leaving_flags.begin(), leaving_flags.end(), 0);
      if (track_deferred_deletion) {
        const size_t required_size =
            static_cast<size_t>(linked_cell->_total_atoms_num);
        if (deferred_mask->size() < required_size) {
          deferred_mask->resize(required_size, 0);
        }
        if (local_marked_in_call.size() != native_atom_size) {
          local_marked_in_call.resize(native_atom_size);
          thrust::fill(local_marked_in_call.begin(),
                       local_marked_in_call.end(), 0);
        }
      }

      // 将单个PositionInfo拷贝到GPU
      PositionInfo* d_pos_info;
      MALLOC(&d_pos_info, sizeof(PositionInfo));
      MEMCPY(d_pos_info, &position_infos[info_idx], sizeof(PositionInfo), H2D);

      mark_leaving_atoms_in_region_op(
          thrust::raw_pointer_cast(_device_data->_d_px.data()),
          thrust::raw_pointer_cast(_device_data->_d_py.data()),
          thrust::raw_pointer_cast(_device_data->_d_pz.data()),
          d_pos_info,
          native_atoms, thrust::raw_pointer_cast(leaving_flags.data()));

      FREE(d_pos_info); // Must free here as we allocate new one in Pass 2 or re-allocate

      if (track_deferred_deletion && native_atom_size > 0) {
        auto mask_begin = deferred_mask->begin();
        thrust::transform(
            leaving_flags.begin(), leaving_flags.end(), mask_begin,
            leaving_flags.begin(),
            [] __device__(int new_mark, int already_marked) {
              return already_marked ? 0 : new_mark;
            });
        // 同一 partner 的多个 region 之间也要去重。
        thrust::transform(
            leaving_flags.begin(), leaving_flags.end(),
            local_marked_in_call.begin(),
            leaving_flags.begin(),
            [] __device__(int new_mark, int already_marked) {
              return already_marked ? 0 : new_mark;
            });
      }

      const int leaving_num_p =
          thrust::reduce(leaving_flags.begin(), leaving_flags.end());
      
      if (leaving_num_p > 0) {
        // Cache indices for this region
        thrust::device_vector<int> indices(leaving_num_p);
        thrust::copy_if(thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(CheckedIntCount(
                            native_atom_size, "leaving native_atom_size")),
                        leaving_flags.begin(),
                        indices.begin(),
                        thrust::identity<int>());
        
        cached_regions.push_back({std::move(indices), leaving_num_p, info_idx});
        total_leaving_atoms += leaving_num_p;

        // 更新本调用内的去重标记；全局 deferred 标记在 sendlist 生成后按
        // “实际发送列表”统一回填，避免“标记了但没发”的分叉。
        if (track_deferred_deletion && native_atom_size > 0) {
           auto mask_begin = local_marked_in_call.begin();
           thrust::transform(
               mask_begin, mask_begin + native_atom_size, leaving_flags.begin(),
               mask_begin, [] __device__(int existing, int new_mark) {
                 return existing | new_mark;
               });
        }
      }
    }

    int current_rank_for_log = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank_for_log);
    AppendTrackedNativeCellStateRows(test_current_step, current_rank_for_log,
                                     _rank, box, linked_cell,
                                     local_marked_in_call, position_infos);

    // Pass 2: full 走 topology exchange；charge/atomic 保持 ref 的旧 SoA 布局。
    if (total_leaving_atoms > 0) {
      if ("full" == atom_style) {
        this->_send_buf.ResizeLeavingExchange(total_leaving_atoms,
                                              this->_send_buf._size_exchange);

        thrust::device_vector<int> sendlist(total_leaving_atoms);
        int current_offset = 0;
        for (const auto& region : cached_regions) {
          int count = region.count;
          if (count > 0) {
            thrust::copy(region.indices.begin(), region.indices.end(),
                         sendlist.begin() + current_offset);
            current_offset += count;
          }
        }
        if (track_deferred_deletion && deferred_mask && !sendlist.empty()) {
          thrust::device_vector<int> sent_marks(total_leaving_atoms, 1);
          thrust::scatter(sent_marks.begin(), sent_marks.end(),
                          sendlist.begin(), deferred_mask->begin());
        }

        thrust::device_vector<rbmd::Real> d_shift_x(total_leaving_atoms, 0);
        thrust::device_vector<rbmd::Real> d_shift_y(total_leaving_atoms, 0);
        thrust::device_vector<rbmd::Real> d_shift_z(total_leaving_atoms, 0);
        {
          int shift_offset = 0;
          for (const auto& region : cached_regions) {
            int count = region.count;
            if (count == 0) continue;
            const auto& pi = position_infos[region.info_idx];
            thrust::fill(d_shift_x.begin() + shift_offset,
                         d_shift_x.begin() + shift_offset + count,
                         pi._shift[0]);
            thrust::fill(d_shift_y.begin() + shift_offset,
                         d_shift_y.begin() + shift_offset + count,
                         pi._shift[1]);
            thrust::fill(d_shift_z.begin() + shift_offset,
                         d_shift_z.begin() + shift_offset + count,
                         pi._shift[2]);
            shift_offset += count;
          }
        }

        op::TopologyPackParams pack_params{};
        pack_params.nsend = total_leaving_atoms;
        pack_params.sendlist = thrust::raw_pointer_cast(sendlist.data());
        pack_params.buf = thrust::raw_pointer_cast(
            this->_send_buf._leaving_atoms_double.data());
        pack_params.size_exchange = this->_send_buf._size_exchange;
        const int pack_nmax =
            (_device_data->nmax > 0)
                ? CheckedIntCount(_device_data->nmax, "topology pack nmax")
                : CheckedIntCount(
                      static_cast<rbmd::Id>(_device_data->_d_atoms_id.size()),
                      "topology pack atom capacity");
        if (pack_nmax <= 0) {
          throw std::runtime_error(
              "Invalid topology pack nmax (<=0), cannot pack leaving atoms.");
        }
        pack_params.nmax = pack_nmax;
        pack_params.d_px = thrust::raw_pointer_cast(_device_data->_d_px.data());
        pack_params.d_py = thrust::raw_pointer_cast(_device_data->_d_py.data());
        pack_params.d_pz = thrust::raw_pointer_cast(_device_data->_d_pz.data());
        pack_params.d_vx = thrust::raw_pointer_cast(_device_data->_d_vx.data());
        pack_params.d_vy = thrust::raw_pointer_cast(_device_data->_d_vy.data());
        pack_params.d_vz = thrust::raw_pointer_cast(_device_data->_d_vz.data());
        pack_params.d_id =
            thrust::raw_pointer_cast(_device_data->_d_atoms_id.data());
        pack_params.d_type =
            thrust::raw_pointer_cast(_device_data->_d_atoms_type.data());
        pack_params.d_molecule =
            thrust::raw_pointer_cast(_device_data->_d_molecular_id.data());
        pack_params.d_charge =
            thrust::raw_pointer_cast(_device_data->_d_charge.data());
        pack_params.d_num_bond =
            thrust::raw_pointer_cast(_device_data->d_num_bond.data());
        pack_params.d_bond_type =
            thrust::raw_pointer_cast(_device_data->d_bond_type.data());
        pack_params.d_bond_atom =
            thrust::raw_pointer_cast(_device_data->d_bond_atom.data());
        pack_params.bond_per_atom = _device_data->bond_per_atom;
        pack_params.d_num_angle =
            thrust::raw_pointer_cast(_device_data->d_num_angle.data());
        pack_params.d_angle_type =
            thrust::raw_pointer_cast(_device_data->d_angle_type.data());
        pack_params.d_angle_atom1 =
            thrust::raw_pointer_cast(_device_data->d_angle_atom1.data());
        pack_params.d_angle_atom2 =
            thrust::raw_pointer_cast(_device_data->d_angle_atom2.data());
        pack_params.d_angle_atom3 =
            thrust::raw_pointer_cast(_device_data->d_angle_atom3.data());
        pack_params.angle_per_atom = _device_data->angle_per_atom;
        pack_params.d_num_dihedral =
            thrust::raw_pointer_cast(_device_data->d_num_dihedral.data());
        pack_params.d_dihedral_type =
            thrust::raw_pointer_cast(_device_data->d_dihedral_type.data());
        pack_params.d_dihedral_atom1 =
            thrust::raw_pointer_cast(_device_data->d_dihedral_atom1.data());
        pack_params.d_dihedral_atom2 =
            thrust::raw_pointer_cast(_device_data->d_dihedral_atom2.data());
        pack_params.d_dihedral_atom3 =
            thrust::raw_pointer_cast(_device_data->d_dihedral_atom3.data());
        pack_params.d_dihedral_atom4 =
            thrust::raw_pointer_cast(_device_data->d_dihedral_atom4.data());
        pack_params.dihedral_per_atom = _device_data->dihedral_per_atom;
        pack_params.d_num_improper =
            thrust::raw_pointer_cast(_device_data->d_num_improper.data());
        pack_params.d_improper_type =
            thrust::raw_pointer_cast(_device_data->d_improper_type.data());
        pack_params.d_improper_atom1 =
            thrust::raw_pointer_cast(_device_data->d_improper_atom1.data());
        pack_params.d_improper_atom2 =
            thrust::raw_pointer_cast(_device_data->d_improper_atom2.data());
        pack_params.d_improper_atom3 =
            thrust::raw_pointer_cast(_device_data->d_improper_atom3.data());
        pack_params.d_improper_atom4 =
            thrust::raw_pointer_cast(_device_data->d_improper_atom4.data());
        pack_params.improper_per_atom = _device_data->improper_per_atom;
        pack_params.d_nspecial =
            thrust::raw_pointer_cast(_device_data->d_nspecial.data());
        pack_params.d_special =
            thrust::raw_pointer_cast(_device_data->d_special.data());
        pack_params.maxspecial = _device_data->maxspecial;
        pack_params.d_shift_x = thrust::raw_pointer_cast(d_shift_x.data());
        pack_params.d_shift_y = thrust::raw_pointer_cast(d_shift_y.data());
        pack_params.d_shift_z = thrust::raw_pointer_cast(d_shift_z.data());
        pack_params.box_min[0] = box->_coord_min[0];
        pack_params.box_min[1] = box->_coord_min[1];
        pack_params.box_min[2] = box->_coord_min[2];
        pack_params.box_max[0] = box->_coord_max[0];
        pack_params.box_max[1] = box->_coord_max[1];
        pack_params.box_max[2] = box->_coord_max[2];

        op::PackExchangeTopologyOp<device::DEVICE_GPU> pack_op;
        pack_op(pack_params);
      } else {
        op::ProcessLeavingAtomOp<device::DEVICE_GPU> process_leaving_atom_op;
        this->_send_buf.ResizeLeaving(total_leaving_atoms);

        const size_t id_size = sizeof(rbmd::Id);
        const size_t real_size = sizeof(rbmd::Real);
        const size_t total_count = total_leaving_atoms;
        size_t global_offset_id = 0;
        size_t global_offset_type = global_offset_id + total_count * id_size;
        size_t global_offset_mol = global_offset_type + total_count * id_size;
        size_t global_offset_px =
            ("full" == atom_style) ? (global_offset_mol + total_count * id_size)
                                   : global_offset_mol;
        size_t global_offset_py = global_offset_px + total_count * real_size;
        size_t global_offset_pz = global_offset_py + total_count * real_size;
        size_t global_offset_vx = global_offset_pz + total_count * real_size;
        size_t global_offset_vy = global_offset_vx + total_count * real_size;
        size_t global_offset_vz = global_offset_vy + total_count * real_size;
        size_t global_offset_charge = global_offset_vz + total_count * real_size;

        char* p_buffer_base =
            thrust::raw_pointer_cast(this->_send_buf._leaving_atoms.data());
        size_t current_atom_offset = 0;
        for (const auto& region : cached_regions) {
          int count = region.count;
          if (count == 0) continue;

          {
            thrust::device_ptr<rbmd::Id> dest_ptr(
                reinterpret_cast<rbmd::Id*>(p_buffer_base + global_offset_id) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_atoms_id.begin(), dest_ptr);
          }
          {
            thrust::device_ptr<rbmd::Id> dest_ptr(
                reinterpret_cast<rbmd::Id*>(p_buffer_base + global_offset_type) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_atoms_type.begin(), dest_ptr);
          }
          if ("full" == atom_style) {
            thrust::device_ptr<rbmd::Id> dest_ptr(
                reinterpret_cast<rbmd::Id*>(p_buffer_base + global_offset_mol) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_molecular_id.begin(), dest_ptr);
          }
          {
            thrust::device_ptr<rbmd::Real> dest_ptr(
                reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_px) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_px.begin(), dest_ptr);
          }
          {
            thrust::device_ptr<rbmd::Real> dest_ptr(
                reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_py) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_py.begin(), dest_ptr);
          }
          {
            thrust::device_ptr<rbmd::Real> dest_ptr(
                reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_pz) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_pz.begin(), dest_ptr);
          }
          {
            thrust::device_ptr<rbmd::Real> dest_ptr(
                reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_vx) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_vx.begin(), dest_ptr);
          }
          {
            thrust::device_ptr<rbmd::Real> dest_ptr(
                reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_vy) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_vy.begin(), dest_ptr);
          }
          {
            thrust::device_ptr<rbmd::Real> dest_ptr(
                reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_vz) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_vz.begin(), dest_ptr);
          }
          if ("charge" == atom_style || "full" == atom_style) {
            thrust::device_ptr<rbmd::Real> dest_ptr(
                reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_charge) +
                current_atom_offset);
            thrust::gather(region.indices.begin(), region.indices.end(),
                           _device_data->_d_charge.begin(), dest_ptr);
          }

          PositionInfo* d_pos_info;
          MALLOC(&d_pos_info, sizeof(PositionInfo));
          MEMCPY(d_pos_info, &position_infos[region.info_idx], sizeof(PositionInfo),
                 H2D);
          rbmd::Real* d_p_px =
              reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_px) +
              current_atom_offset;
          rbmd::Real* d_p_py =
              reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_py) +
              current_atom_offset;
          rbmd::Real* d_p_pz =
              reinterpret_cast<rbmd::Real*>(p_buffer_base + global_offset_pz) +
              current_atom_offset;
          process_leaving_atom_op(d_p_px, d_p_py, d_p_pz, d_pos_info,
                                  static_cast<rbmd::Id>(count), *box);
          FREE(d_pos_info);
          current_atom_offset += count;
        }

        if (track_deferred_deletion && deferred_mask) {
          auto mask_begin = deferred_mask->begin();
          for (const auto& region : cached_regions) {
            if (region.count <= 0) {
              continue;
            }
            thrust::device_vector<int> sent_marks(region.count, 1);
            thrust::scatter(sent_marks.begin(), sent_marks.end(),
                            region.indices.begin(), mask_begin);
          }
        }
      }
    }
  }

  // ========== 阶段 2: 处理 Halo 粒子 ==========
  if (msgType == HALO_COPIES || msgType == LEAVING_AND_HALO_COPIES) {
    // 根据文档3.2.3节的三阶段通信方案：
    // - 第1阶段（x维）：处理 [0, native_num)
    // - 第2阶段（y维）：处理 [0, native_num + x维halo)
    // - 第3阶段（z维）：处理 [0, native_num + x维halo + y维halo)
    // end_atom_idx 表示当前已累积的总原子数（包括native和之前阶段收到的halo）
    
    rbmd::Id actual_start = 0;  // 总是从0开始
    rbmd::Id actual_end = (end_atom_idx > 0) ? end_atom_idx : linked_cell->_total_atoms_num;
    
    if (msgType == LEAVING_AND_HALO_COPIES) {
      std::cerr << "ERROR: Cannot collect halo after removing leaving atoms in same pass!" << std::endl;
      return;
    }
    
    // 调整flags大小到实际需要处理的范围
    rbmd::Id range_size = actual_end - actual_start;
    if (range_size <= 0) {
      return; // 没有原子需要处理
    }
    const bool trace_halo_decision = ShouldTraceHaloDecision();
    const auto& tracked_gids = GetHaloDecisionTrackedGids();
    
    halo_flags.resize(range_size);
    thrust::fill(halo_flags.begin(), halo_flags.end(), 0);
    thrust::device_vector<int> halo_owner(range_size);
    thrust::fill(halo_owner.begin(), halo_owner.end(), -1);
    
    // 创建偏移后的指针，指向要处理的原子范围
    rbmd::Real* px_offset = thrust::raw_pointer_cast(_device_data->_d_px.data()) + actual_start;
    rbmd::Real* py_offset = thrust::raw_pointer_cast(_device_data->_d_py.data()) + actual_start;
    rbmd::Real* pz_offset = thrust::raw_pointer_cast(_device_data->_d_pz.data()) + actual_start;
    
    // 调试输出
    int current_rank = 100;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
// #ifndef NDEBUG
//     if (current_rank == 7) {  // 只输出rank 7的信息用于调试
//       std::cout << "[DEBUG Rank " << current_rank << "] Processing halo atoms:" << std::endl;
//       std::cout << "  - Range: [" << actual_start << ", " << actual_end << ")" << std::endl;
//       std::cout << "  - Range size: " << range_size << std::endl;
//       std::cout << "  - Num PositionInfos: " << position_infos.size() << std::endl;
//       for (size_t i = 0; i < position_infos.size(); ++i) {
//         std::cout << "  - PositionInfo[" << i << "] copies: "
//                   << "[" << position_infos[i]._copiesLow[0] << ", " << position_infos[i]._copiesLow[1] << ", " << position_infos[i]._copiesLow[2] << "] to "
//                   << "[" << position_infos[i]._copiesHigh[0] << ", " << position_infos[i]._copiesHigh[1] << ", " << position_infos[i]._copiesHigh[2] << "]"
//                   << std::endl;
//       }
//     }
// #endif

    op::MarkHaloAtomsInRegionOp<device::DEVICE_GPU> mark_halo_atoms_in_region_op;

    // 逐个处理每个PositionInfo
    for (size_t info_idx = 0; info_idx < position_infos.size(); ++info_idx) {
      // 将单个PositionInfo拷贝到GPU
      PositionInfo* d_pos_info;
      MALLOC(&d_pos_info, sizeof(PositionInfo));
      MEMCPY(d_pos_info, &position_infos[info_idx], sizeof(PositionInfo), H2D);

      // 标记在copies区域中的原子
      mark_halo_atoms_in_region_op(
          px_offset,
          py_offset,
          pz_offset,
          d_pos_info,
          range_size, // 使用实际范围大小
          thrust::raw_pointer_cast(halo_flags.data()),
          thrust::raw_pointer_cast(halo_owner.data()),
          static_cast<int>(info_idx));

      CHECK_RUNTIME(FREE(d_pos_info));
    }

    std::vector<int> h_raw_halo_marks;
    std::vector<int> h_leaving_mask;
    std::vector<rbmd::Id> h_ids_in_range;
    std::vector<rbmd::Real> h_px_in_range;
    std::vector<rbmd::Real> h_py_in_range;
    std::vector<rbmd::Real> h_pz_in_range;
    if (trace_halo_decision && !tracked_gids.empty()) {
      const size_t host_range_size = static_cast<size_t>(range_size);
      h_raw_halo_marks.resize(host_range_size);
      thrust::copy(halo_flags.begin(), halo_flags.end(), h_raw_halo_marks.begin());

      h_ids_in_range.resize(host_range_size);
      h_px_in_range.resize(host_range_size);
      h_py_in_range.resize(host_range_size);
      h_pz_in_range.resize(host_range_size);
      thrust::copy(_device_data->_d_atoms_id.begin() + actual_start,
                   _device_data->_d_atoms_id.begin() + actual_end,
                   h_ids_in_range.begin());
      thrust::copy(_device_data->_d_px.begin() + actual_start,
                   _device_data->_d_px.begin() + actual_end,
                   h_px_in_range.begin());
      thrust::copy(_device_data->_d_py.begin() + actual_start,
                   _device_data->_d_py.begin() + actual_end,
                   h_py_in_range.begin());
      thrust::copy(_device_data->_d_pz.begin() + actual_start,
                   _device_data->_d_pz.begin() + actual_end,
                   h_pz_in_range.begin());
    }
    if (trace_halo_decision && !tracked_gids.empty()) {
      std::vector<int> h_final_halo_marks(static_cast<size_t>(range_size));
      thrust::copy(halo_flags.begin(), halo_flags.end(), h_final_halo_marks.begin());
      int current_rank_for_log = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &current_rank_for_log);
      AppendHaloDecisionRows(
          test_current_step, current_rank_for_log, _rank, actual_start, actual_end,
          linked_cell->_native_atoms_num, linked_cell->_ghost_atoms_num,
          linked_cell->_total_atoms_num, tracked_gids, h_ids_in_range,
          h_px_in_range, h_py_in_range, h_pz_in_range, h_raw_halo_marks,
          h_final_halo_marks, h_leaving_mask);
    }

    int halo_num = thrust::reduce(halo_flags.begin(), halo_flags.end());

// #ifndef NDEBUG
//     if (current_rank == 7) {
//       std::cout << "  - Marked halo atoms: " << halo_num << std::endl;
//     }
// #endif

    // 调整ghost缓冲区大小
    this->_send_buf.ResizeGhost(halo_num);

    if (halo_num > 0) {
      // 打包Halo粒子
      const size_t id_size = sizeof(rbmd::Id);
      const size_t real_size = sizeof(rbmd::Real);
      char* p_ghost_buffer = thrust::raw_pointer_cast(
          _send_buf._ghost_atoms.data());

      size_t current_offset = 0;

      // 拷贝 ID
      {
        auto target_raw_ptr = reinterpret_cast<rbmd::Id*>(
          p_ghost_buffer + current_offset);
        thrust::device_ptr<rbmd::Id> target_iterator(target_raw_ptr);
        thrust::copy_if(_device_data->_d_atoms_id.begin() + actual_start,
                        _device_data->_d_atoms_id.begin() + actual_end,
                        halo_flags.begin(),
                        target_iterator,
                        thrust::identity<int>());
      }
      current_offset += halo_num * id_size;

      // 拷贝 type
      {
        auto target_raw_ptr = reinterpret_cast<rbmd::Id*>(
          p_ghost_buffer + current_offset);
        thrust::device_ptr<rbmd::Id> target_iterator(target_raw_ptr);
        thrust::copy_if(_device_data->_d_atoms_type.begin() + actual_start,
                        _device_data->_d_atoms_type.begin() + actual_end,
                        halo_flags.begin(),
                        target_iterator,
                        thrust::identity<int>());
      }
      current_offset += halo_num * id_size;

      // 拷贝坐标 px, py, pz
      {
        auto target_raw_ptr = reinterpret_cast<rbmd::Real*>(
          p_ghost_buffer + current_offset);
        thrust::device_ptr<rbmd::Real> target_iterator(target_raw_ptr);
        thrust::copy_if(_device_data->_d_px.begin() + actual_start,
                        _device_data->_d_px.begin() + actual_end,
                        halo_flags.begin(),
                        target_iterator,
                        thrust::identity<int>());
      }
      current_offset += halo_num * real_size;

      {
        auto target_raw_ptr = reinterpret_cast<rbmd::Real*>(
          p_ghost_buffer + current_offset);
        thrust::device_ptr<rbmd::Real> target_iterator(target_raw_ptr);
        thrust::copy_if(_device_data->_d_py.begin() + actual_start,
                        _device_data->_d_py.begin() + actual_end,
                        halo_flags.begin(),
                        target_iterator,
                        thrust::identity<int>());
      }
      current_offset += halo_num * real_size;

      {
        auto target_raw_ptr = reinterpret_cast<rbmd::Real*>(
          p_ghost_buffer + current_offset);
        thrust::device_ptr<rbmd::Real> target_iterator(target_raw_ptr);
        thrust::copy_if(_device_data->_d_pz.begin() + actual_start,
                        _device_data->_d_pz.begin() + actual_end,
                        halo_flags.begin(),
                        target_iterator,
                        thrust::identity<int>());
      }
      current_offset += halo_num * real_size;

      // 拷贝电荷（如果需要）
      if ("charge" == atom_style || "full" == atom_style) {
        {
          auto target_raw_ptr = reinterpret_cast<rbmd::Real*>(
            p_ghost_buffer + current_offset);
          thrust::device_ptr<rbmd::Real> target_iterator(target_raw_ptr);
          thrust::copy_if(_device_data->_d_charge.begin() + actual_start,
                          _device_data->_d_charge.begin() + actual_end,
                          halo_flags.begin(),
                          target_iterator,
                          thrust::identity<int>());
        }
      }

      thrust::device_vector<int> halo_atom_owner_indices(halo_num);
      thrust::copy_if(
          halo_owner.begin(), halo_owner.end(),
          halo_flags.begin(),
          halo_atom_owner_indices.begin(),
          thrust::identity<int>());

      // 对发送缓冲区中的halo数据应用周期性边界处理
      char* p_ghost_buffer_k = thrust::raw_pointer_cast(
          _send_buf._ghost_atoms.data());

      size_t current_offset_k = 0;

      // 跳过ID和type
      current_offset_k += halo_num * id_size * 2;

      // 获取坐标指针
      rbmd::Real* d_p_px = reinterpret_cast<rbmd::Real*>(
        p_ghost_buffer_k + current_offset_k);
      current_offset_k += halo_num * real_size;
      rbmd::Real* d_p_py = reinterpret_cast<rbmd::Real*>(
        p_ghost_buffer_k + current_offset_k);
      current_offset_k += halo_num * real_size;
      rbmd::Real* d_p_pz = reinterpret_cast<rbmd::Real*>(
        p_ghost_buffer_k + current_offset_k);
      int* d_halo_owner_indices =
          thrust::raw_pointer_cast(halo_atom_owner_indices.data());

      // 逐个处理Halo原子的周期性边界条件
      op::ProcessHaloAtomOp<device::DEVICE_GPU> process_halo_atom_op;
      for (size_t info_idx = 0; info_idx < position_infos.size(); ++info_idx) {
        // 将单个PositionInfo拷贝到GPU
        PositionInfo* d_pos_info;
        MALLOC(&d_pos_info, sizeof(PositionInfo));
        MEMCPY(d_pos_info, &position_infos[info_idx], sizeof(PositionInfo), H2D);

        process_halo_atom_op(d_p_px, d_p_py, d_p_pz,
                            d_pos_info,
                            halo_num, *box,
                            d_halo_owner_indices,
                            static_cast<int>(info_idx));

        FREE(d_pos_info);
      }
    }
  }
}

void CommunicationPartner::ResetRecvStatus() {
  _leavingReceivedDone = false;
  _haloReceivedDone = false;
  _isReceiving = false;
  _countLeavingTested = 0;
  _countHaloTested = 0;
}

void CommunicationPartner::deadlockDiagnosticSend() const {
  if (not _msgSent and _isSending) {
    std::cout << "Send request to " << _rank << " not yet completed"
        << std::endl;
  }
}

/**
 * @brief (Updated) Diagnoses deadlock on the receiving side.
 * Checks if any posted receives are stuck waiting for completion.
 */
void CommunicationPartner::deadlockDiagnosticRecv() const {
  // Check if we are stuck waiting for a "leaving" message to finish receiving.
  if (_isLeavingRecvPosted && !_isLeavingRecvFinalized) {
    std::cout << "DIAGNOSTIC: Rank is waiting to complete RECEIVE of LEAVING message from " << _rank << std::endl;
  }

  // Check if we are stuck waiting for a "halo" message to finish receiving.
  if (_isHaloRecvPosted && !_isHaloRecvFinalized) {
    std::cout << "DIAGNOSTIC: Rank is waiting to complete RECEIVE of HALO message from " << _rank << std::endl;
  }
}

/**
 * @brief (Updated) Diagnoses deadlock on both sending and receiving sides.
 */
void CommunicationPartner::deadlockDiagnosticSendRecv() const {
  // NOTE: You should also review 'deadlockDiagnosticSend()' to ensure
  // it correctly uses the '_msgSent' and '_isSending' flags.
  deadlockDiagnosticSend();

  // The updated Recv diagnostic provides all necessary receive-side information.
  deadlockDiagnosticRecv();
}

void CommunicationPartner::OutputLeavingAtomsToCSV(const std::string& filename,
                                                   int current_rank,
                                                   rbmd::Id timestep) {
  // 计算接收到的leaving原子数量
  unsigned int recv_leaving_atoms_num = 0;

  if (_recv_buf._leaving_size > 0) {
    recv_leaving_atoms_num = _recv_buf._leaving_atoms.size() / _recv_buf._leaving_size;
  }
  if (recv_leaving_atoms_num == 0) {
    std::cout << "[Rank " << current_rank << "] No leaving atoms to output. Returning early." << std::endl;
    return;
  }

  // 解析接收缓冲区中的原子数据
  std::vector<char> recv_atoms_array;
  recv_atoms_array.resize(_recv_buf._leaving_atoms.size());
  thrust::copy(_recv_buf._leaving_atoms.begin(),_recv_buf._leaving_atoms.end(),recv_atoms_array.begin());
  char* p_recv_buffer = recv_atoms_array.data();
  const size_t id_size = sizeof(rbmd::Id);
  const size_t real_size = sizeof(rbmd::Real);

  auto atom_style = DataManager::getInstance().getConfigData()->Get<std::string>(
      "atom_style", "init_configuration", "read_data");

  // 创建原子信息映射，以ID为key进行排序
  std::map<rbmd::Id, std::vector<std::string>> atom_info_map;

  // 缓冲区按属性顺序存储（SoA），依次为 ID、type、[molecule_id]、坐标、速度、[电荷]

  size_t offset_block = 0;
  std::vector<rbmd::Id> ids(recv_leaving_atoms_num);
  std::memcpy(ids.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * id_size);
  offset_block += recv_leaving_atoms_num * id_size;

  std::vector<rbmd::Id> types(recv_leaving_atoms_num);
  std::memcpy(types.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * id_size);
  offset_block += recv_leaving_atoms_num * id_size;

  std::vector<rbmd::Id> molecule_ids;
  if ("full" == atom_style) {
    molecule_ids.resize(recv_leaving_atoms_num);
    std::memcpy(molecule_ids.data(), p_recv_buffer + offset_block,
                recv_leaving_atoms_num * id_size);
    offset_block += recv_leaving_atoms_num * id_size;
  }

  std::vector<rbmd::Real> px(recv_leaving_atoms_num);
  std::memcpy(px.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * real_size);
  offset_block += recv_leaving_atoms_num * real_size;

  std::vector<rbmd::Real> py(recv_leaving_atoms_num);
  std::memcpy(py.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * real_size);
  offset_block += recv_leaving_atoms_num * real_size;

  std::vector<rbmd::Real> pz(recv_leaving_atoms_num);
  std::memcpy(pz.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * real_size);
  offset_block += recv_leaving_atoms_num * real_size;

  std::vector<rbmd::Real> vx(recv_leaving_atoms_num);
  std::memcpy(vx.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * real_size);
  offset_block += recv_leaving_atoms_num * real_size;

  std::vector<rbmd::Real> vy(recv_leaving_atoms_num);
  std::memcpy(vy.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * real_size);
  offset_block += recv_leaving_atoms_num * real_size;

  std::vector<rbmd::Real> vz(recv_leaving_atoms_num);
  std::memcpy(vz.data(), p_recv_buffer + offset_block,
              recv_leaving_atoms_num * real_size);
  offset_block += recv_leaving_atoms_num * real_size;

  std::vector<rbmd::Real> charges;
  if ("full" == atom_style || "charge" == atom_style) {
    charges.resize(recv_leaving_atoms_num);
    std::memcpy(charges.data(), p_recv_buffer + offset_block,
                recv_leaving_atoms_num * real_size);
    offset_block += recv_leaving_atoms_num * real_size;
  }

  for (unsigned int i = 0; i < recv_leaving_atoms_num; ++i) {
    std::vector<std::string> atom_info;
    atom_info.push_back(std::to_string(ids[i]));
    atom_info.push_back(std::to_string(types[i]));
    if ("full" == atom_style) {
      atom_info.push_back(std::to_string(molecule_ids[i]));
    }
    atom_info.push_back(std::to_string(px[i]));
    atom_info.push_back(std::to_string(py[i]));
    atom_info.push_back(std::to_string(pz[i]));
    atom_info.push_back(std::to_string(vx[i]));
    atom_info.push_back(std::to_string(vy[i]));
    atom_info.push_back(std::to_string(vz[i]));
    if (!charges.empty()) {
      atom_info.push_back(std::to_string(charges[i]));
    }
    atom_info_map[ids[i]] = std::move(atom_info);
  }

  // 打开CSV文件进行追加写入
  std::ofstream csv_file;
  bool file_exists = std::ifstream(filename).good();
  csv_file.open(filename, std::ios::app);

  if (!csv_file.is_open()) {
    std::cerr << "[Rank " << current_rank << "] ERROR: Unable to open CSV file " << filename << " for writing" << std::endl;
    return;
  }

  std::cout << "[Rank " << current_rank << "] Successfully opened CSV file: " << filename
            << " (file_exists=" << (file_exists ? "true" : "false") << ")" << std::endl;

  // 如果文件不存在，写入表头
  if (!file_exists) {
    csv_file << "id,type";
    if ("full" == atom_style) {
      csv_file << ",molecule_id";
    }
    csv_file << ",px,py,pz,vx,vy,vz";
    if ("full" == atom_style || "charge" == atom_style) {
      csv_file << ",charge";
    }
    csv_file << std::endl;
  }

  // 按ID排序后写入数据
  std::cout << "[Rank " << current_rank << "] Writing " << atom_info_map.size() << " leaving atoms to CSV file." << std::endl;
  for (const auto& pair : atom_info_map) {
    for (const std::string& info : pair.second) {
      csv_file << info;
      if (&info != &pair.second.back()) {
        csv_file << ",";
      }
    }
    csv_file << std::endl;
  }

  csv_file.close();
  std::cout << "[Rank " << current_rank << "] Successfully wrote leaving atoms to CSV file: " << filename << std::endl;
}

void CommunicationPartner::OutputGhostAtomsToCSV(const std::string& filename,
                                                 int current_rank,
                                                 rbmd::Id timestep) {
  // 计算接收到的ghost原子数量
  unsigned int recv_ghost_atoms_num = 0;

  if (_recv_buf._ghost_size > 0) {
    recv_ghost_atoms_num = _recv_buf._ghost_atoms.size() / _recv_buf._ghost_size;
  }

  if (recv_ghost_atoms_num == 0) {
    return;
  }

  // 解析接收缓冲区中的原子数据
  std::vector<char> recv_atoms_array;
  recv_atoms_array.resize(_recv_buf._ghost_atoms.size());
  thrust::copy(_recv_buf._ghost_atoms.begin(),_recv_buf._ghost_atoms.end(),recv_atoms_array.begin());
  char* p_recv_buffer = recv_atoms_array.data();
  const size_t id_size = sizeof(rbmd::Id);
  const size_t real_size = sizeof(rbmd::Real);

  auto atom_style = DataManager::getInstance().getConfigData()->Get<std::string>(
      "atom_style", "init_configuration", "read_data");

  // 创建原子信息映射，以ID为key进行排序
  std::map<rbmd::Id, std::vector<std::string>> atom_info_map;

  // 缓冲区采用 SoA 布局：依次存放 ID、type、坐标、[电荷]

  size_t offset_block = 0;
  std::vector<rbmd::Id> ids(recv_ghost_atoms_num);
  std::memcpy(ids.data(), p_recv_buffer + offset_block,
              recv_ghost_atoms_num * id_size);
  offset_block += recv_ghost_atoms_num * id_size;

  std::vector<rbmd::Id> types(recv_ghost_atoms_num);
  std::memcpy(types.data(), p_recv_buffer + offset_block,
              recv_ghost_atoms_num * id_size);
  offset_block += recv_ghost_atoms_num * id_size;

  std::vector<rbmd::Real> px(recv_ghost_atoms_num);
  std::memcpy(px.data(), p_recv_buffer + offset_block,
              recv_ghost_atoms_num * real_size);
  offset_block += recv_ghost_atoms_num * real_size;

  std::vector<rbmd::Real> py(recv_ghost_atoms_num);
  std::memcpy(py.data(), p_recv_buffer + offset_block,
              recv_ghost_atoms_num * real_size);
  offset_block += recv_ghost_atoms_num * real_size;

  std::vector<rbmd::Real> pz(recv_ghost_atoms_num);
  std::memcpy(pz.data(), p_recv_buffer + offset_block,
              recv_ghost_atoms_num * real_size);
  offset_block += recv_ghost_atoms_num * real_size;

  std::vector<rbmd::Real> charges;
  if ("charge" == atom_style || "full" == atom_style) {
    charges.resize(recv_ghost_atoms_num);
    std::memcpy(charges.data(), p_recv_buffer + offset_block,
                recv_ghost_atoms_num * real_size);
    offset_block += recv_ghost_atoms_num * real_size;
  }

  for (unsigned int i = 0; i < recv_ghost_atoms_num; ++i) {
    std::vector<std::string> atom_info;
    atom_info.push_back(std::to_string(ids[i]));
    atom_info.push_back(std::to_string(types[i]));
    atom_info.push_back(std::to_string(px[i]));
    atom_info.push_back(std::to_string(py[i]));
    atom_info.push_back(std::to_string(pz[i]));
    if (!charges.empty()) {
      atom_info.push_back(std::to_string(charges[i]));
    }
    atom_info_map[ids[i]] = std::move(atom_info);
  }

  // 打开CSV文件进行追加写入
  std::ofstream csv_file;
  bool file_exists = std::ifstream(filename).good();
  csv_file.open(filename, std::ios::app);

  if (!csv_file.is_open()) {
    std::cerr << "Error: Unable to open CSV file " << filename << " for writing" << std::endl;
    return;
  }

  // 如果文件不存在，写入表头
  if (!file_exists) {
    csv_file << "id,type,px,py,pz";
    if ("charge" == atom_style || "full" == atom_style) {
      csv_file << ",charge";
    }
    csv_file << std::endl;
  }

  // 按ID排序后写入数据
  for (const auto& pair : atom_info_map) {
    for (const std::string& info : pair.second) {
      csv_file << info;
      if (&info != &pair.second.back()) {
        csv_file << ",";
      }
    }
    csv_file << std::endl;
  }

  csv_file.close();
}

void CommunicationPartner::outputHaloInfoToFile(int current_rank, int total_rank, const MPI_Comm& comm) {
    // 将 _haloInfo 数据格式化为字符串
    std::stringstream ss;
    ss << "=================================================\n";
    ss << "           进程 " << current_rank << " 的 HaloInfo\n";
    ss << "=================================================\n";
    
    for (size_t i = 0; i < _haloInfo.size(); ++i) {
        const auto &info = _haloInfo[i];
        ss << "PositionInfo [" << i << "]:\n";
        
        // 输出 leaving 区域
        ss << "  Leaving Region:\n";
        ss << "    Low : x=" << std::fixed << std::setprecision(6) << info._leavingLow[0]
           << ", y=" << info._leavingLow[1]
           << ", z=" << info._leavingLow[2] << "\n";
        ss << "    High: x=" << info._leavingHigh[0]
           << ", y=" << info._leavingHigh[1]
           << ", z=" << info._leavingHigh[2] << "\n";
        
        // 输出 copies 区域
        ss << "  Copies Region:\n";
        ss << "    Low : x=" << info._copiesLow[0]
           << ", y=" << info._copiesLow[1]
           << ", z=" << info._copiesLow[2] << "\n";
        ss << "    High: x=" << info._copiesHigh[0]
           << ", y=" << info._copiesHigh[1]
           << ", z=" << info._copiesHigh[2] << "\n";
        
        // 输出 shift 值
        ss << "  Shift: x=" << info._shift[0]
           << ", y=" << info._shift[1]
           << ", z=" << info._shift[2] << "\n";
        
        // 输出 offset 值
        ss << "  Offset: x=" << info._offset[0]
           << ", y=" << info._offset[1]
           << ", z=" << info._offset[2] << "\n";
        
        ss << "------------------------------------------------\n";
    }
    
    std::string data_str = ss.str();
    int data_length = data_str.length();
    
    if (current_rank == 0) {
        // Rank 0 进程：接收所有其他进程的数据并写入文件
        std::ofstream outfile("halo_info_all_processes.txt");
        if (outfile.is_open()) {
            outfile << "=================================================\n";
            outfile << "           所有进程 HaloInfo 汇总\n";
            outfile << "           总进程数: " << total_rank << "\n";
            outfile << "=================================================\n\n";
            
            // 先写入自己的数据
            outfile << data_str;
            
            // 接收其他进程的数据
            for (int src_rank = 1; src_rank < total_rank; ++src_rank) {
                int recv_length;
                MPI_Status status;
                
                // 接收字符串长度
                MPI_Recv(&recv_length, 1, MPI_INT, src_rank, 0, comm, &status);
                
                // 接收字符串内容
                std::vector<char> recv_data(recv_length + 1);
                MPI_Recv(recv_data.data(), recv_length, MPI_CHAR, src_rank, 1, comm, &status);
                recv_data[recv_length] = '\0';
                
                outfile << std::string(recv_data.data());
            }
            
            outfile.close();
            std::cout << "HaloInfo 信息已成功写入 halo_info_all_processes.txt 文件" << std::endl;
        } else {
            std::cerr << "无法打开文件 halo_info_all_processes.txt 进行写入" << std::endl;
        }
    } else {
        // 非 rank 0 进程：发送数据给 rank 0
        
        // 发送字符串长度
        MPI_Send(&data_length, 1, MPI_INT, 0, 0, comm);
        
        // 发送字符串内容
        MPI_Send(data_str.c_str(), data_length, MPI_CHAR, 0, 1, comm);
    }
    
    // 确保所有进程都完成这个函数后再继续
    MPI_Barrier(comm);
}
