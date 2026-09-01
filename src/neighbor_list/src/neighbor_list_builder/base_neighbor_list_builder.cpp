
#include "neighbor_list_builder/base_neighbor_list_builder.h"

#include <sys/stat.h>
#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "common/device_types.h"
#include "common/mpi_root_guard.hpp"
#include "common/rbmd_define.h"
#include "data_manager.h"
#include "force_box_selector.h"
#include "linked_cell/linked_cell_locator.h"
#include "neighbor_list_op.h"

#ifdef USE_MPI
#include "mpi.h"
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

struct NeighborSkinCacheDebugConfig {
  bool enabled{false};
  int every{1};
};

const NeighborSkinCacheDebugConfig& GetNeighborSkinCacheDebugConfig() {
  static const NeighborSkinCacheDebugConfig cfg{
      EnvFlagEnabled("RBMD_DEBUG_NEIGHBOR_SKIN"),
      EnvIntOrDefault("RBMD_DEBUG_NEIGHBOR_SKIN_EVERY", 1)};
  return cfg;
}

bool ShouldWriteNeighborSkinCacheDebug(rbmd::Id step) {
  const auto& cfg = GetNeighborSkinCacheDebugConfig();
  return cfg.enabled && cfg.every > 0 && (step % cfg.every) == 0;
}

void EnsureNeighborSkinCacheDebugDirectory() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  (void)mkdir("logs", 0755);
  (void)mkdir("logs/debug", 0755);
  initialized = true;
}

int CurrentProcessId() { return static_cast<int>(::getpid()); }

bool BoxGeometryEqual(const Box& lhs, const Box& rhs) {
  if (lhs._type != rhs._type || lhs._pbc_x != rhs._pbc_x ||
      lhs._pbc_y != rhs._pbc_y || lhs._pbc_z != rhs._pbc_z) {
    return false;
  }
  for (int d = 0; d < 3; ++d) {
    if (lhs._coord_min[d] != rhs._coord_min[d] ||
        lhs._coord_max[d] != rhs._coord_max[d] ||
        lhs._length[d] != rhs._length[d]) {
      return false;
    }
  }
  return true;
}

struct DisplacementSq {
  const rbmd::Real* px;
  const rbmd::Real* py;
  const rbmd::Real* pz;
  const rbmd::Real* ref_px;
  const rbmd::Real* ref_py;
  const rbmd::Real* ref_pz;
  Box box;

  __host__ __device__ rbmd::Real operator()(rbmd::Id idx) const {
    rbmd::Real dx = px[idx] - ref_px[idx];
    rbmd::Real dy = py[idx] - ref_py[idx];
    rbmd::Real dz = pz[idx] - ref_pz[idx];
    MinImageDistance_while(box, dx, dy, dz);
    return dx * dx + dy * dy + dz * dz;
  }
};

}  // namespace

BaseNeighborListBuilder::BaseNeighborListBuilder() {
  this->_box = DataManager::getInstance().getMDData()->_box;
  this->_linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  CHECK_RUNTIME(MALLOC(&_d_should_realloc, sizeof(rbmd::Id)));
}
BaseNeighborListBuilder::~BaseNeighborListBuilder() {
  CHECK_RUNTIME(FREE(_d_should_realloc));
}

void BaseNeighborListBuilder::ReductionSum(rbmd::Id* d_src_array,
                                           rbmd::Id* d_dst, rbmd::Id size) {
  auto first = thrust::device_pointer_cast(d_src_array);
  auto last = first + size;
  *thrust::device_pointer_cast(d_dst) =
      thrust::reduce(first, last, rbmd::Id{0});
}

void BaseNeighborListBuilder::InitNeighborListIndices() {
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif

  thrust::exclusive_scan(
      _neighbor_list->_d_max_neighbor_num.begin(),
      _neighbor_list->_d_max_neighbor_num.begin() + active_atoms,
      _neighbor_list->_start_idx.begin());
  op::InitEndIndexOp<device::DEVICE_GPU> init_end_index_op;
  init_end_index_op(
      thrust::raw_pointer_cast(_neighbor_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(_neighbor_list->_end_idx.data()), active_atoms);
  // 这里索引没有问题 2024-09-12
}

void BaseNeighborListBuilder::ValidateCellGridSupportsPBC() const {
  if (_linked_cell->_total_cells >= _neighbor_cell_num) {
    return;
  }

  std::ostringstream oss;
  oss << "The current simulation domain is too small for PBC to be effective: "
      << "_linked_cell->_total_cells=" << _linked_cell->_total_cells
      << ", _neighbor_cell_num=" << _neighbor_cell_num
      << ", cutoff=" << _linked_cell->_cutoff << ", cell_count_within_cutoff="
      << _linked_cell->_cell_count_within_cutoff
      << ". Increase the box size, reduce the cutoff, or use fewer MPI ranks.";
  throw std::runtime_error(oss.str());
}

bool BaseNeighborListBuilder::ShouldUseStrictMPIGridValidation() const {
#ifdef USE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return false;
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) {
    return false;
  }
  int total_ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
  return total_ranks > 1;
#else
  return false;
#endif
}

bool BaseNeighborListBuilder::CanReuseNeighborList() const {
  if (_linked_cell == nullptr || _linked_cell->_skin <= rbmd::Real(0)) {
    return false;
  }

#ifdef USE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return false;
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) {
    return false;
  }
#endif
  return true;
}

rbmd::Id BaseNeighborListBuilder::ActiveAtomsForNeighborCache() const {
  rbmd::Id active_atoms = _linked_cell ? _linked_cell->_total_atoms_num : 0;
#ifdef USE_MPI
  active_atoms = _linked_cell ? _linked_cell->_native_atoms_num : 0;
#endif
  return std::max<rbmd::Id>(active_atoms, 0);
}

bool BaseNeighborListBuilder::ShouldReuseNeighborList(
    rbmd::Real neighbor_cutoff) {
  _last_max_displacement_sq = rbmd::Real(-1);
  _last_rebuild_threshold_sq = rbmd::Real(-1);
  _last_cache_reason = "not_checked";

  if (!CanReuseNeighborList()) {
    _last_cache_reason = "disabled";
    return false;
  }
  const rbmd::Id active_atoms = ActiveAtomsForNeighborCache();
  const auto device_data = DataManager::getInstance().getDeviceData();
  const rbmd::Real threshold = _linked_cell->_skin * rbmd::Real(0.5);
  _last_rebuild_threshold_sq = threshold * threshold;
  bool local_cache_valid = true;
  auto invalidate = [&](const char* reason) {
    if (local_cache_valid) {
      _last_cache_reason = reason;
    }
    local_cache_valid = false;
  };

  if (!_has_neighbor_cache) invalidate("no_cache");
  if (should_realloc) invalidate("should_realloc");
  if (active_atoms <= 0) invalidate("no_active_atoms");
  if (active_atoms != _cached_active_atoms) invalidate("active_atoms_changed");
  if (_linked_cell->_total_atoms_num != _cached_total_atoms) {
    invalidate("total_atoms_changed");
  }
  if (std::abs(neighbor_cutoff - _cached_neighbor_cutoff) > EPSILON) {
    invalidate("neighbor_cutoff_changed");
  }
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (!_cached_force_box_valid ||
      !BoxGeometryEqual(force_box, _cached_force_box)) {
    invalidate("box_geometry_changed");
  }

  const auto active_size =
      static_cast<std::size_t>(std::max<rbmd::Id>(0, active_atoms));
  const auto total_size = static_cast<std::size_t>(
      std::max<rbmd::Id>(0, _linked_cell->_total_atoms_num));
  if (!device_data || _cached_px.size() < active_size ||
      _cached_py.size() < active_size || _cached_pz.size() < active_size ||
      _cached_atom_ids.size() < total_size ||
      (device_data && device_data->_d_atoms_id.size() < total_size)) {
    invalidate("cache_vector_too_small");
  }
  if (local_cache_valid &&
      !thrust::equal(device_data->_d_atoms_id.begin(),
                     device_data->_d_atoms_id.begin() + total_size,
                     _cached_atom_ids.begin())) {
    invalidate("atom_order_changed");
  }

  rbmd::Real max_displacement_sq = rbmd::Real(0);
  if (local_cache_valid) {
    max_displacement_sq = thrust::transform_reduce(
        thrust::device, thrust::make_counting_iterator<rbmd::Id>(0),
        thrust::make_counting_iterator<rbmd::Id>(active_atoms),
        DisplacementSq{thrust::raw_pointer_cast(device_data->_d_px.data()),
                       thrust::raw_pointer_cast(device_data->_d_py.data()),
                       thrust::raw_pointer_cast(device_data->_d_pz.data()),
                       thrust::raw_pointer_cast(_cached_px.data()),
                       thrust::raw_pointer_cast(_cached_py.data()),
                       thrust::raw_pointer_cast(_cached_pz.data()), force_box},
        rbmd::Real(0), thrust::maximum<rbmd::Real>());
  }
  _last_max_displacement_sq = max_displacement_sq;

#ifdef USE_MPI
  int local_rebuild =
      (!local_cache_valid || max_displacement_sq > _last_rebuild_threshold_sq)
          ? 1
          : 0;
  int global_rebuild = 0;
  rbmd::Real global_max_displacement_sq = rbmd::Real(0);
  MPI_CHECK(MPI_Allreduce(&local_rebuild, &global_rebuild, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allreduce(&max_displacement_sq, &global_max_displacement_sq, 1,
                          MPI_RBMD_REAL, MPI_MAX, MPI_COMM_WORLD));
  _last_max_displacement_sq = global_max_displacement_sq;
  if (global_rebuild == 0) {
#else
  if (local_cache_valid && max_displacement_sq <= _last_rebuild_threshold_sq) {
#endif
    _last_cache_reason = "within_skin";
    ++_neighbor_cache_reuse_count;
    EmitNeighborSkinCacheDebug("reuse", neighbor_cutoff, active_atoms);
    return true;
  }

  if (local_cache_valid && max_displacement_sq > _last_rebuild_threshold_sq) {
    _last_cache_reason = "displacement_exceeded";
#ifdef USE_MPI
  } else if (local_cache_valid) {
    _last_cache_reason = "remote_rebuild_required";
#endif
  }
  return false;
}

void BaseNeighborListBuilder::RecordNeighborBuildState(
    rbmd::Real neighbor_cutoff) {
  const rbmd::Id active_atoms = ActiveAtomsForNeighborCache();
  const auto device_data = DataManager::getInstance().getDeviceData();
  const rbmd::Id total_atoms =
      _linked_cell ? _linked_cell->_total_atoms_num : 0;
  bool local_ready = CanReuseNeighborList() && !should_realloc &&
                     active_atoms > 0 && total_atoms >= active_atoms &&
                     device_data != nullptr;

  if (local_ready) {
    const auto size = static_cast<std::size_t>(active_atoms);
    const auto total_size = static_cast<std::size_t>(total_atoms);
    local_ready = device_data->_d_px.size() >= size &&
                  device_data->_d_py.size() >= size &&
                  device_data->_d_pz.size() >= size &&
                  device_data->_d_atoms_id.size() >= total_size;
    if (local_ready) {
      _cached_px.resize(size);
      _cached_py.resize(size);
      _cached_pz.resize(size);
      _cached_atom_ids.resize(total_size);
      thrust::copy_n(device_data->_d_px.begin(), active_atoms,
                     _cached_px.begin());
      thrust::copy_n(device_data->_d_py.begin(), active_atoms,
                     _cached_py.begin());
      thrust::copy_n(device_data->_d_pz.begin(), active_atoms,
                     _cached_pz.begin());
      thrust::copy_n(device_data->_d_atoms_id.begin(), total_atoms,
                     _cached_atom_ids.begin());

      // 三阶段计划以稳定 ID 固化排序后的索引。重复 ghost ID 时回退重建，
      // 避免把同一 ID 的不同周期镜像错误合并。
      thrust::host_vector<rbmd::Id> host_ids(total_size);
      thrust::copy(_cached_atom_ids.begin(), _cached_atom_ids.end(),
                   host_ids.begin());
      std::unordered_set<rbmd::Id> unique_ids;
      unique_ids.reserve(total_size);
      for (const auto id : host_ids) {
        if (!unique_ids.insert(id).second) {
          local_ready = false;
          break;
        }
      }
    }
  }

#ifdef USE_MPI
  int local_ready_int = local_ready ? 1 : 0;
  int global_ready_int = 0;
  MPI_CHECK(MPI_Allreduce(&local_ready_int, &global_ready_int, 1, MPI_INT,
                          MPI_MIN, MPI_COMM_WORLD));
  local_ready = global_ready_int != 0;
#endif
  if (!local_ready) {
    _has_neighbor_cache = false;
    _cached_force_box_valid = false;
    return;
  }

  _cached_active_atoms = active_atoms;
  _cached_total_atoms = total_atoms;
  _cached_neighbor_cutoff = neighbor_cutoff;
  _cached_force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  _cached_force_box_valid = true;
  _has_neighbor_cache = true;
  ++_neighbor_cache_rebuild_count;
  EmitNeighborSkinCacheDebug("rebuild", neighbor_cutoff, active_atoms);
}

void BaseNeighborListBuilder::EmitNeighborSkinCacheDebug(
    const char* action, rbmd::Real neighbor_cutoff,
    rbmd::Id active_atoms) const {
  if (!ShouldWriteNeighborSkinCacheDebug(test_current_step)) {
    return;
  }

  EnsureNeighborSkinCacheDebugDirectory();
  const int rank = rbmd::mpi::CurrentRank();
  const int pid = CurrentProcessId();
  const std::string filename = "logs/debug/neighbor_skin_cache_rank" +
                               std::to_string(rank) + "_pid" +
                               std::to_string(pid) + ".csv";
  std::ofstream csv(filename, std::ios::app);
  if (!csv.is_open()) {
    return;
  }

  if (csv.tellp() == 0) {
    csv << "step,rank,action,reason,active_atoms,skin,neighbor_cutoff,"
           "max_displacement_sq,rebuild_threshold_sq,reuse_count,"
           "rebuild_count\n";
  }

  csv << test_current_step << "," << rank << "," << action << ","
      << _last_cache_reason << "," << active_atoms << ","
      << (_linked_cell ? _linked_cell->_skin : rbmd::Real(0)) << ","
      << neighbor_cutoff << "," << _last_max_displacement_sq << ","
      << _last_rebuild_threshold_sq << "," << _neighbor_cache_reuse_count << ","
      << _neighbor_cache_rebuild_count << "\n";
}
