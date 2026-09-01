#include "linked_cell/linked_cell.h"


#include "../common/rbmd_define.h"
#include "../common/device_types.h"
#include "../common/types.h"
#include   "../data_manager/include/model/md_data.h"
#include "common/neighbor_skin.h"
#include "data_manager.h"
#include "linked_cell_op.h"
#include "model/md_data.h"
#include "common/mpi_root_guard.hpp"
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>

#ifdef USE_MPI
#include "rbmd_parallel_until_locator.h"
#endif

extern rbmd::Id test_current_step;

namespace {
template <typename T>
void ResizeVector(thrust::device_vector<T>& vec, rbmd::Id size) {
  vec.resize(static_cast<std::size_t>(std::max<rbmd::Id>(size, 0)));
}

struct RowMajorPermIndex {
  const int* perm;
  int width;

  __host__ __device__ std::size_t operator()(std::size_t e) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int new_i = static_cast<int>(e / static_cast<std::size_t>(width));
    const int k = static_cast<int>(
        e - static_cast<std::size_t>(new_i) * static_cast<std::size_t>(width));
    const int old_i = perm[new_i];
    return static_cast<std::size_t>(old_i) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(k);
#else
    (void)e;
    return 0;
#endif
  }
};

bool NativeOwnershipInvariantAuditEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_NATIVE_OWNERSHIP_INVARIANTS");
  if (env == nullptr) {
    return false;
  }
  return env[0] != '\0' && std::string(env) != "0";
}

void EnsureNativeOwnershipInvariantDirectory() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  (void)mkdir("logs", 0755);
  (void)mkdir("logs/debug", 0755);
  initialized = true;
}

rbmd::Real OutsideLocalSubdomainDistance(const Box& local_box,
                                         rbmd::Real px,
                                         rbmd::Real py,
                                         rbmd::Real pz) {
  rbmd::Real dx = rbmd::Real(0);
  rbmd::Real dy = rbmd::Real(0);
  rbmd::Real dz = rbmd::Real(0);
  if (px < local_box._coord_min[0]) {
    dx = local_box._coord_min[0] - px;
  } else if (px >= local_box._coord_max[0]) {
    dx = px - local_box._coord_max[0];
  }
  if (py < local_box._coord_min[1]) {
    dy = local_box._coord_min[1] - py;
  } else if (py >= local_box._coord_max[1]) {
    dy = py - local_box._coord_max[1];
  }
  if (pz < local_box._coord_min[2]) {
    dz = local_box._coord_min[2] - pz;
  } else if (pz >= local_box._coord_max[2]) {
    dz = pz - local_box._coord_max[2];
  }
  return MAX(dx, MAX(dy, dz));
}

void AppendNativeOwnershipInvariantPostAssignRow(LinkedCell* linked_cell) {
  auto device_data = DataManager::getInstance().getDeviceData();
  auto local_box = DataManager::getInstance().getMDData()->_box;
  if (!NativeOwnershipInvariantAuditEnabled() || linked_cell == nullptr ||
      !device_data || !local_box) {
    return;
  }

  const rbmd::Id native_atoms = linked_cell->_native_atoms_num;
  if (native_atoms <= 0) {
    return;
  }

  const std::size_t native_n = static_cast<std::size_t>(native_atoms);
  std::vector<rbmd::Id> h_ids(native_n, 0);
  std::vector<rbmd::Real> h_px(native_n, 0);
  std::vector<rbmd::Real> h_py(native_n, 0);
  std::vector<rbmd::Real> h_pz(native_n, 0);
  std::vector<rbmd::Id> h_cell_ids(native_n, -1);
  std::vector<Cell> h_cells(linked_cell->_cells.size());

  thrust::copy(device_data->_d_atoms_id.begin(),
               device_data->_d_atoms_id.begin() + native_atoms,
               h_ids.begin());
  thrust::copy(device_data->_d_px.begin(),
               device_data->_d_px.begin() + native_atoms,
               h_px.begin());
  thrust::copy(device_data->_d_py.begin(),
               device_data->_d_py.begin() + native_atoms,
               h_py.begin());
  thrust::copy(device_data->_d_pz.begin(),
               device_data->_d_pz.begin() + native_atoms,
               h_pz.begin());
  thrust::copy(linked_cell->_per_atom_cell_id.begin(),
               linked_cell->_per_atom_cell_id.begin() + native_atoms,
               h_cell_ids.begin());
  if (!h_cells.empty()) {
    thrust::copy(linked_cell->_cells.begin(), linked_cell->_cells.end(),
                 h_cells.begin());
  }

  int native_outside_local_subdomain_count = 0;
  int native_in_halo_cell_count = 0;
  int shell_exceeded_count = 0;
  rbmd::Real max_outside_distance = rbmd::Real(0);
  rbmd::Id first_outside_gid = -1;
  rbmd::Id first_halo_gid = -1;
  rbmd::Id first_shell_exceeded_gid = -1;

  for (std::size_t i = 0; i < native_n; ++i) {
    const rbmd::Real px = h_px[i];
    const rbmd::Real py = h_py[i];
    const rbmd::Real pz = h_pz[i];
    const bool in_local_subdomain =
        local_box->_coord_min[0] <= px &&
        px < local_box->_coord_max[0] &&
        local_box->_coord_min[1] <= py &&
        py < local_box->_coord_max[1] &&
        local_box->_coord_min[2] <= pz &&
        pz < local_box->_coord_max[2];
    const rbmd::Real outside_distance =
        OutsideLocalSubdomainDistance(*local_box, px, py, pz);
    if (outside_distance > max_outside_distance) {
      max_outside_distance = outside_distance;
    }
    if (!in_local_subdomain) {
      ++native_outside_local_subdomain_count;
      if (first_outside_gid < 0) {
        first_outside_gid = h_ids[i];
      }
      if (outside_distance > linked_cell->_halo_cutoff) {
        ++shell_exceeded_count;
        if (first_shell_exceeded_gid < 0) {
          first_shell_exceeded_gid = h_ids[i];
        }
      }
    }

    const rbmd::Id cell_idx = h_cell_ids[i];
    const bool cell_is_halo =
        cell_idx >= 0 && static_cast<std::size_t>(cell_idx) < h_cells.size() &&
        h_cells[static_cast<std::size_t>(cell_idx)]._is_halo;
    if (cell_is_halo) {
      ++native_in_halo_cell_count;
      if (first_halo_gid < 0) {
        first_halo_gid = h_ids[i];
      }
    }
  }

  EnsureNativeOwnershipInvariantDirectory();
  const int current_rank = rbmd::mpi::CurrentRank();
  std::ofstream csv("logs/debug/native_ownership_invariants_rank" +
                        std::to_string(current_rank) + "_pid" +
                        std::to_string(static_cast<int>(::getpid())) + ".csv",
                    std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,phase,rank,native_atoms,native_outside_local_subdomain_count,"
           "native_in_halo_cell_count,shell_exceeded_count,max_outside_distance,"
           "halo_cutoff,first_outside_gid,first_halo_gid,"
           "first_shell_exceeded_gid\n";
  }

  csv << test_current_step << ",after_assign_atoms_to_cell," << current_rank
      << "," << native_atoms << "," << native_outside_local_subdomain_count
      << "," << native_in_halo_cell_count << "," << shell_exceeded_count
      << "," << max_outside_distance << "," << linked_cell->_halo_cutoff
      << "," << first_outside_gid << "," << first_halo_gid << ","
      << first_shell_exceeded_gid << "\n";
}
}  // namespace

LinkedCell::LinkedCell() {
  this->_box = DataManager::getInstance().getMDData()->_box;
  this->_device_data = DataManager::getInstance().getDeviceData();
  this->_structure_info_data =
      DataManager::getInstance().getMDData()->_structure_info_data;
  this->_config_data = DataManager::getInstance().getConfigData();
  this->_per_atom_cell_id.resize(*(_structure_info_data->_num_atoms));
  // TODO  反序列化
  this->_cutoff =
      _config_data->Get<rbmd::Real>("cut_off", "hyper_parameters", "neighbor");
  this->_skin = rbmd::neighbor::ReadSkinOrDefault(_config_data.get());
  this->_neighbor_cutoff =
      rbmd::neighbor::NeighborCutoff(this->_cutoff, this->_skin, "LinkedCell");
  this->_total_atoms_num =
      *(_structure_info_data->_num_atoms);  // do this because nativate num
  this->_native_atoms_num = this->_total_atoms_num;
  this->_ghost_atoms_num = 0;
#ifdef USE_MPI
  const auto global_atoms = GET_RBMD_PARALLEL->_global_structure_info.total_atoms;
  this->_atom_id_to_idx.resize(static_cast<std::size_t>(
      std::max<rbmd::Id>(global_atoms, this->_total_atoms_num)));
#else
  this->_atom_id_to_idx.resize(static_cast<std::size_t>(this->_total_atoms_num));
#endif
}

LinkedCell::~LinkedCell() {
  CHECK_RUNTIME(FREE(this->_linked_cell_device_data_ptr));
}

__host__ void LinkedCell::Build() {
  const auto per_cell_length =
      _neighbor_cutoff / static_cast<rbmd::Real>(_cell_count_within_cutoff);
  if (!std::isfinite(static_cast<double>(per_cell_length)) ||
      per_cell_length <= rbmd::Real(0)) {
    std::ostringstream oss;
    oss << "[rank " << rbmd::mpi::CurrentRank()
        << "] LinkedCell::Build invalid per_cell_length: cutoff=" << _cutoff
        << ", skin=" << _skin
        << ", neighbor_cutoff=" << _neighbor_cutoff
        << ", cell_count_within_cutoff=" << _cell_count_within_cutoff
        << ", per_cell_length=" << per_cell_length;
    throw std::runtime_error(oss.str());
  }

#ifdef USE_MPI
  if (_halo_cutoff == rbmd::Real(0)) {
    _halo_cutoff = _neighbor_cutoff;
  }
  rbmd::neighbor::ValidateHaloCutoffCoversNeighborCutoff(
      _halo_cutoff, _neighbor_cutoff, "LinkedCell::Build");
#endif

  std::size_t total_cells = 1;
  for (int dim = 0; dim < 3; dim++) {
    const rbmd::Real diff = _box->_coord_max[dim] - _box->_coord_min[dim];
    if (!std::isfinite(static_cast<double>(diff)) || diff <= rbmd::Real(0)) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LinkedCell::Build invalid local box extent at dim " << dim
          << ": coord_min=" << _box->_coord_min[dim]
          << ", coord_max=" << _box->_coord_max[dim]
          << ", diff=" << diff;
      throw std::runtime_error(oss.str());
    }

    const double native_cells_double =
        std::floor(static_cast<double>(diff) /
                   static_cast<double>(per_cell_length));
    if (!std::isfinite(native_cells_double) || native_cells_double < 1.0) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LinkedCell::Build invalid native cell count at dim " << dim
          << ": diff=" << diff
          << ", per_cell_length=" << per_cell_length
          << ", native_cells_double=" << native_cells_double
          << ", cutoff=" << _cutoff
          << ", neighbor_cutoff=" << _neighbor_cutoff
          << ", halo_cutoff=" << _halo_cutoff;
      throw std::runtime_error(oss.str());
    }

    _box->_box_width_as_cell_units[dim] =
        static_cast<rbmd::Id>(native_cells_double);
    _nativate_cells[dim] = _box->_box_width_as_cell_units[dim];

    _cell_length[dim] = diff / _box->_box_width_as_cell_units[dim];
    if (!std::isfinite(static_cast<double>(_cell_length[dim])) ||
        _cell_length[dim] <= rbmd::Real(0)) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LinkedCell::Build invalid cell_length at dim " << dim
          << ": diff=" << diff
          << ", native_cells=" << _box->_box_width_as_cell_units[dim]
          << ", cell_length=" << _cell_length[dim];
      throw std::runtime_error(oss.str());
    }

    rbmd::Id halo_cells = 0;
#ifdef USE_MPI
    const double halo_cells_double =
        std::ceil(static_cast<double>(_halo_cutoff) /
                  static_cast<double>(_cell_length[dim]));
    if (!std::isfinite(halo_cells_double) || halo_cells_double < 0.0) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LinkedCell::Build invalid halo cell count at dim " << dim
          << ": halo_cutoff=" << _halo_cutoff
          << ", cell_length=" << _cell_length[dim]
          << ", halo_cells_double=" << halo_cells_double;
      throw std::runtime_error(oss.str());
    }
    halo_cells = static_cast<rbmd::Id>(halo_cells_double);
#endif
    _box->_halo_width_as_cell_units[dim] = halo_cells;
    _per_dimension_cells[dim] = _nativate_cells[dim] + 2 * halo_cells;
    if (_per_dimension_cells[dim] == 2 * halo_cells) {
      throw std::runtime_error("The simulation domain is too small to use MPI.");
    }
    if (_per_dimension_cells[dim] <= 0) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LinkedCell::Build non-positive per_dimension_cells at dim " << dim
          << ": native_cells=" << _nativate_cells[dim]
          << ", halo_cells=" << halo_cells
          << ", per_dimension_cells=" << _per_dimension_cells[dim];
      throw std::runtime_error(oss.str());
    }
    constexpr std::size_t kMaxReasonableCellsPerDim = 1u << 20;
    if (static_cast<std::size_t>(_per_dimension_cells[dim]) >
        kMaxReasonableCellsPerDim) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LinkedCell::Build suspiciously large per_dimension_cells at dim "
          << dim << ": native_cells=" << _nativate_cells[dim]
          << ", halo_cells=" << halo_cells
          << ", per_dimension_cells=" << _per_dimension_cells[dim]
          << ", cutoff=" << _cutoff
          << ", neighbor_cutoff=" << _neighbor_cutoff
          << ", halo_cutoff=" << _halo_cutoff
          << ", box_extent=" << diff;
      throw std::runtime_error(oss.str());
    }
    if (total_cells >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(_per_dimension_cells[dim])) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LinkedCell::Build total_cells overflow before dim " << dim
          << ": running_total_cells=" << total_cells
          << ", per_dimension_cells=" << _per_dimension_cells[dim];
      throw std::runtime_error(oss.str());
    }
    total_cells *= static_cast<std::size_t>(_per_dimension_cells[dim]);

    _halo_length[dim] = halo_cells * _cell_length[dim];
    _box->_halo_coord_min[dim] = _box->_coord_min[dim] - _halo_length[dim];
    _box->_halo_coord_max[dim] = _box->_coord_max[dim] + _halo_length[dim];
    _nativate_cell_start[dim] = halo_cells;
    _nativate_cell_end[dim] = _per_dimension_cells[dim] - halo_cells;
    _cell_length_reciprocal[dim] = 1.0 / _cell_length[dim];
  }
  constexpr std::size_t kMaxReasonableTotalCells = 300000000;
  if (total_cells > kMaxReasonableTotalCells ||
      total_cells >
          static_cast<std::size_t>(std::numeric_limits<rbmd::Id>::max())) {
    std::ostringstream oss;
    oss << "[rank " << rbmd::mpi::CurrentRank()
        << "] LinkedCell::Build suspicious total_cells=" << total_cells
        << " with per_dimension_cells=(" << _per_dimension_cells[0] << ", "
        << _per_dimension_cells[1] << ", " << _per_dimension_cells[2] << ")"
        << ", cutoff=" << _cutoff
        << ", neighbor_cutoff=" << _neighbor_cutoff
        << ", halo_cutoff=" << _halo_cutoff
        << ", local_box_extent=("
        << (_box->_coord_max[0] - _box->_coord_min[0]) << ", "
        << (_box->_coord_max[1] - _box->_coord_min[1]) << ", "
        << (_box->_coord_max[2] - _box->_coord_min[2]) << ")";
    throw std::runtime_error(oss.str());
  }

  this->_total_cells = static_cast<rbmd::Id>(total_cells);
  this->_in_atom_list_start_index.resize(total_cells);
  this->_in_atom_list_end_index.resize(total_cells);
  this->_cells.resize(total_cells);

  if (_device_data->_d_box == nullptr) {
    CHECK_RUNTIME(MALLOC(&_device_data->_d_box, sizeof(Box)));
  }
  CHECK_RUNTIME(MEMCPY(_device_data->_d_box, _box.get(), sizeof(Box), H2D));
}

LinkedCellDeviceDataPtr* LinkedCell::GetDataPtr() {
  if (nullptr == this->_linked_cell_device_data_ptr) {
    this->AllocDeviceMemory();
    this->SyncHToD();
  }
  return this->_linked_cell_device_data_ptr;
}

void LinkedCell::InitializeCells() {
  op::InitializeCellOp<device::DEVICE_GPU> initialize_cell_op;
  initialize_cell_op(GetDataPtr(), _device_data->_d_box,
                     thrust::raw_pointer_cast(this->_cells.data()),
                     this->_total_cells);
}

void LinkedCell::AssignAtomsToCell() {
  if (_per_atom_cell_id.size() < static_cast<std::size_t>(_total_atoms_num)) {
    _per_atom_cell_id.resize(static_cast<std::size_t>(_total_atoms_num));
  }
  op::AssignAtomsToCellOp<device::DEVICE_GPU> assign_atoms_to_cell_op;
  assign_atoms_to_cell_op(thrust::raw_pointer_cast(_device_data->_d_px.data()),
                          thrust::raw_pointer_cast(_device_data->_d_py.data()),
                          thrust::raw_pointer_cast(_device_data->_d_pz.data()),
                          _device_data->_d_box, GetDataPtr(),
                          thrust::raw_pointer_cast(this->_cells.data()),
                          thrust::raw_pointer_cast(_per_atom_cell_id.data()),
                          this->_total_atoms_num);
  AppendNativeOwnershipInvariantPostAssignRow(this);
}

void LinkedCell::ComputeCellRangesIndices() {
  thrust::fill(_in_atom_list_start_index.begin(), _in_atom_list_start_index.end(),
               rbmd::Id(0));
  thrust::fill(_in_atom_list_end_index.begin(), _in_atom_list_end_index.end(),
               rbmd::Id(0));
  op::ComputeCellRangesIndicesOp<device::DEVICE_GPU>
      compute_cell_ranges_indices_op;
  compute_cell_ranges_indices_op(
      thrust::raw_pointer_cast(_cell_sorted_cell_id.data()),
      thrust::raw_pointer_cast(_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_in_atom_list_end_index.data()),
      this->_total_atoms_num);
}

void LinkedCell::SyncHToD() {
  if (nullptr == this->_linked_cell_device_data_ptr) {
    this->AllocDeviceMemory();
  }
  CHECK_RUNTIME(MEMCPY(&_linked_cell_device_data_ptr->_d_total_cells,
                       &this->_total_cells, sizeof(rbmd::Id), H2D));

  CHECK_RUNTIME(MEMCPY(_linked_cell_device_data_ptr->_d_per_dimension_cells,
                       this->_per_dimension_cells, ALIGN_SIZE(rbmd::Id, 3),
                       H2D));

  CHECK_RUNTIME(MEMCPY(_linked_cell_device_data_ptr->_d_cell_length,
                       this->_cell_length, ALIGN_SIZE(rbmd::Real, 3), H2D));

  CHECK_RUNTIME(MEMCPY(_linked_cell_device_data_ptr->_d_cell_length_reciprocal,
                       this->_cell_length_reciprocal, ALIGN_SIZE(rbmd::Real, 3),
                       H2D));

  CHECK_RUNTIME(MEMCPY(&_linked_cell_device_data_ptr->_d_cutoff,
                       &this->_neighbor_cutoff, sizeof(rbmd::Real), H2D));
  CHECK_RUNTIME(MEMCPY(_linked_cell_device_data_ptr->_d_nativate_cells_start,
                       this->_nativate_cell_start, ALIGN_SIZE(rbmd::Id, 3), H2D));
  CHECK_RUNTIME(MEMCPY(_linked_cell_device_data_ptr->_d_nativate_cells_dims,
                       this->_nativate_cells, ALIGN_SIZE(rbmd::Id, 3), H2D));
  CHECK_RUNTIME(MEMCPY(_linked_cell_device_data_ptr->_d_covers_whole_domain,
                       this->_covers_whole_domain, ALIGN_SIZE(bool, 3), H2D));
}

void LinkedCell::SortAtomsByCellKey() {
  auto atom_style = DataManager::getInstance().getConfigData()->Get<std::string>(
      "atom_style", "init_configuration", "read_data");
  const bool has_charge = (atom_style == "charge" || atom_style == "full");
  const bool has_molecule = (atom_style == "full");

  auto& data = *_device_data;
  const rbmd::Id native_n = _native_atoms_num;
  const rbmd::Id total_n = _total_atoms_num;
  if (native_n < 0 || total_n < 0 || total_n == 0) {
    return;
  }
  if (data._d_atoms_id.size() < static_cast<std::size_t>(total_n) ||
      _per_atom_cell_id.size() < static_cast<std::size_t>(total_n)) {
    throw std::runtime_error(
        "LinkedCell::SortAtomsByCellKey count/vector size mismatch before "
        "MapAtomidToIdx");
  }

  auto reorder_1d_segment = [&](auto& vec, const thrust::device_vector<int>& perm,
                                rbmd::Id dst_start) {
    using Vec = std::decay_t<decltype(vec)>;
    using T = typename Vec::value_type;
    const rbmd::Id len = static_cast<rbmd::Id>(perm.size());
    if (len <= 0) {
      return;
    }
    const size_t need = static_cast<size_t>(dst_start + len);
    if (vec.size() < need) {
      return;
    }
    thrust::device_vector<T> tmp(static_cast<size_t>(len));
    thrust::gather(perm.begin(), perm.end(), vec.begin(), tmp.begin());
    thrust::copy(tmp.begin(), tmp.end(), vec.begin() + dst_start);
  };

  auto reorder_2d_flat_rows = [&](auto& vec,
                                  const thrust::device_vector<int>& perm,
                                  rbmd::Id dst_row_start, rbmd::Id row_count,
                                  int width) {
    using Vec = std::decay_t<decltype(vec)>;
    using T = typename Vec::value_type;
    if (width <= 0 || row_count <= 0) {
      return;
    }
    const size_t out_size =
        static_cast<size_t>(row_count) * static_cast<size_t>(width);
    const size_t dst_offset =
        static_cast<size_t>(dst_row_start) * static_cast<size_t>(width);
    if (vec.size() < dst_offset + out_size) {
      return;
    }
    if (perm.size() < static_cast<size_t>(row_count)) {
      return;
    }

    thrust::device_vector<T> tmp(out_size);
    auto begin = thrust::make_counting_iterator<std::size_t>(0);
    auto map_it = thrust::make_transform_iterator(
        begin, RowMajorPermIndex{thrust::raw_pointer_cast(perm.data()), width});
    thrust::gather(thrust::device, map_it, map_it + out_size, vec.begin(),
                   tmp.begin());
    thrust::copy(tmp.begin(), tmp.end(), vec.begin() + dst_offset);
  };

  if (native_n > 0) {
    thrust::device_vector<int> perm_native(static_cast<size_t>(native_n));
    thrust::sequence(perm_native.begin(), perm_native.end(), 0);
    thrust::stable_sort_by_key(_per_atom_cell_id.begin(),
                               _per_atom_cell_id.begin() + native_n,
                               perm_native.begin());

    reorder_1d_segment(data._d_vx, perm_native, 0);
    reorder_1d_segment(data._d_vy, perm_native, 0);
    reorder_1d_segment(data._d_vz, perm_native, 0);
    reorder_1d_segment(data._d_flagX, perm_native, 0);
    reorder_1d_segment(data._d_flagY, perm_native, 0);
    reorder_1d_segment(data._d_flagZ, perm_native, 0);

    reorder_1d_segment(data._d_atoms_id, perm_native, 0);
    reorder_1d_segment(data._d_atoms_type, perm_native, 0);
    if (has_molecule) {
      reorder_1d_segment(data._d_molecular_id, perm_native, 0);
    }
    reorder_1d_segment(data._d_px, perm_native, 0);
    reorder_1d_segment(data._d_py, perm_native, 0);
    reorder_1d_segment(data._d_pz, perm_native, 0);
    if (has_charge) {
      reorder_1d_segment(data._d_charge, perm_native, 0);
    }

    if (!data.d_num_bond.empty() && data.bond_per_atom > 0) {
      reorder_1d_segment(data.d_num_bond, perm_native, 0);
      reorder_2d_flat_rows(data.d_bond_type, perm_native, 0, native_n,
                           data.bond_per_atom);
      reorder_2d_flat_rows(data.d_bond_atom, perm_native, 0, native_n,
                           data.bond_per_atom);
    }
    if (!data.d_num_angle.empty() && data.angle_per_atom > 0) {
      reorder_1d_segment(data.d_num_angle, perm_native, 0);
      reorder_2d_flat_rows(data.d_angle_type, perm_native, 0, native_n,
                           data.angle_per_atom);
      reorder_2d_flat_rows(data.d_angle_atom1, perm_native, 0, native_n,
                           data.angle_per_atom);
      reorder_2d_flat_rows(data.d_angle_atom2, perm_native, 0, native_n,
                           data.angle_per_atom);
      reorder_2d_flat_rows(data.d_angle_atom3, perm_native, 0, native_n,
                           data.angle_per_atom);
    }
    if (!data.d_num_dihedral.empty() && data.dihedral_per_atom > 0) {
      reorder_1d_segment(data.d_num_dihedral, perm_native, 0);
      reorder_2d_flat_rows(data.d_dihedral_type, perm_native, 0, native_n,
                           data.dihedral_per_atom);
      reorder_2d_flat_rows(data.d_dihedral_atom1, perm_native, 0, native_n,
                           data.dihedral_per_atom);
      reorder_2d_flat_rows(data.d_dihedral_atom2, perm_native, 0, native_n,
                           data.dihedral_per_atom);
      reorder_2d_flat_rows(data.d_dihedral_atom3, perm_native, 0, native_n,
                           data.dihedral_per_atom);
      reorder_2d_flat_rows(data.d_dihedral_atom4, perm_native, 0, native_n,
                           data.dihedral_per_atom);
    }
    if (!data.d_num_improper.empty() && data.improper_per_atom > 0) {
      reorder_1d_segment(data.d_num_improper, perm_native, 0);
      reorder_2d_flat_rows(data.d_improper_type, perm_native, 0, native_n,
                           data.improper_per_atom);
      reorder_2d_flat_rows(data.d_improper_atom1, perm_native, 0, native_n,
                           data.improper_per_atom);
      reorder_2d_flat_rows(data.d_improper_atom2, perm_native, 0, native_n,
                           data.improper_per_atom);
      reorder_2d_flat_rows(data.d_improper_atom3, perm_native, 0, native_n,
                           data.improper_per_atom);
      reorder_2d_flat_rows(data.d_improper_atom4, perm_native, 0, native_n,
                           data.improper_per_atom);
    }
    if (!data.d_nspecial.empty()) {
      reorder_2d_flat_rows(data.d_nspecial, perm_native, 0, native_n, 3);
    }
    if (!data.d_special.empty() && data.maxspecial > 0) {
      reorder_2d_flat_rows(data.d_special, perm_native, 0, native_n,
                           data.maxspecial);
    }
  }

  if (total_n > native_n) {
    const rbmd::Id halo_n = total_n - native_n;
    thrust::device_vector<int> perm_halo(static_cast<size_t>(halo_n));
    thrust::sequence(perm_halo.begin(), perm_halo.end(),
                     static_cast<int>(native_n));
    thrust::stable_sort_by_key(_per_atom_cell_id.begin() + native_n,
                               _per_atom_cell_id.begin() + total_n,
                               perm_halo.begin());

    reorder_1d_segment(data._d_atoms_id, perm_halo, native_n);
    reorder_1d_segment(data._d_atoms_type, perm_halo, native_n);
    if (has_molecule) {
      reorder_1d_segment(data._d_molecular_id, perm_halo, native_n);
    }
    reorder_1d_segment(data._d_px, perm_halo, native_n);
    reorder_1d_segment(data._d_py, perm_halo, native_n);
    reorder_1d_segment(data._d_pz, perm_halo, native_n);
    if (has_charge) {
      reorder_1d_segment(data._d_charge, perm_halo, native_n);
    }
  }

  _cell_sorted_cell_id.resize(static_cast<size_t>(total_n));
  _cell_sorted_atom_indices.resize(static_cast<size_t>(total_n));
  if (total_n > 0) {
    thrust::copy(_per_atom_cell_id.begin(), _per_atom_cell_id.begin() + total_n,
                 _cell_sorted_cell_id.begin());
    thrust::sequence(_cell_sorted_atom_indices.begin(),
                     _cell_sorted_atom_indices.begin() + total_n, 0);
    thrust::stable_sort_by_key(_cell_sorted_cell_id.begin(),
                               _cell_sorted_cell_id.begin() + total_n,
                               _cell_sorted_atom_indices.begin());
  }

  thrust::fill(_atom_id_to_idx.begin(), _atom_id_to_idx.end(), rbmd::Id(-1));
  op::MapAtomidToIdxOp<device::DEVICE_GPU> map_atomid_to_idx_op;
  map_atomid_to_idx_op(thrust::raw_pointer_cast(_atom_id_to_idx.data()),
                       raw_ptr(_device_data->_d_atoms_id), _total_atoms_num);
}


template <typename T>
void LinkedCell::MapAtomId(thrust::device_vector<T>& d_target) {
  auto* d_atom_id_to_idx = thrust::raw_pointer_cast(_atom_id_to_idx.data());
  thrust::transform(d_target.begin(), d_target.end(), d_target.begin(),
                    [d_atom_id_to_idx] __device__(T t) {
                      return static_cast<T>(d_atom_id_to_idx[t]);
                    });
}
template <typename T>
void LinkedCell::MapAtomIdForce(thrust::device_vector<T>& d_target) {
  // 创建一个临时数组来存储结果
  thrust::device_vector<T> d_fx_mapped(d_target.size());

  // 使用 gather 函数重新排列 d_fx
  thrust::gather(_atom_id_to_idx.begin(),  // 索引数组的起始迭代器
                 _atom_id_to_idx.end(),    // 索引数组的结束迭代器
                 d_target.begin(),         // 输入数组的起始迭代器
                 d_fx_mapped.begin()       // 输出数组的起始迭代器
  );

  // 将结果复制回原数组
  d_target = d_fx_mapped;
}

void LinkedCell::AllocDeviceMemory() {
  if (nullptr == this->_linked_cell_device_data_ptr) {
    CHECK_RUNTIME(MALLOC(&(this->_linked_cell_device_data_ptr),
                         sizeof(LinkedCellDeviceDataPtr)));
  }
}

void LinkedCell::UpdateLeavingNum(int change_leaving_num) {
  _native_atoms_num = std::max<rbmd::Id>(0, _native_atoms_num + change_leaving_num);
  if (_structure_info_data && _structure_info_data->_num_atoms) {
    *(_structure_info_data->_num_atoms) = _native_atoms_num;
  }
  _total_atoms_num = _native_atoms_num + _ghost_atoms_num;
}

void LinkedCell::UpdateGhostNum(rbmd::Id change_ghost_num) {
  _ghost_atoms_num = std::max<rbmd::Id>(0, _ghost_atoms_num + change_ghost_num);
  _total_atoms_num = _native_atoms_num + _ghost_atoms_num;
}

void LinkedCell::ResizeDataLeaving(size_t leaving_size) {
  if (leaving_size == 0) {
    return;
  }
  const rbmd::Id target_native =
      _native_atoms_num + static_cast<rbmd::Id>(leaving_size);
  const rbmd::Id target_total = target_native + _ghost_atoms_num;
  ResizeNativeDependentVectors(target_native, target_total);
}

void LinkedCell::EnsureCapacity(rbmd::Id required_total_atoms) {
  if (required_total_atoms <= _total_atoms_num) {
    return;
  }
  ResizePerAtomVectors(required_total_atoms);
}

void LinkedCell::ResizeNativeDependentVectors(rbmd::Id target_native_atoms,
                                              rbmd::Id target_total_atoms) {
  _native_atoms_num = std::max<rbmd::Id>(target_native_atoms, 0);
  _ghost_atoms_num =
      std::max<rbmd::Id>(target_total_atoms - _native_atoms_num, 0);
  _total_atoms_num = _native_atoms_num + _ghost_atoms_num;
  ResizePerAtomVectors(_total_atoms_num);
  if (_structure_info_data && _structure_info_data->_num_atoms) {
    *(_structure_info_data->_num_atoms) = _native_atoms_num;
  }
}

// 谓词结构体，检查对应的 Cell 是否为 halo
struct is_halo_predicate {
  const Cell *cells;

  is_halo_predicate(const Cell *_cells) : cells(_cells) {}

  template <typename Tuple>
  __host__ __device__ bool operator()(const Tuple &t) const {
    // 从元组中提取 cell_idx
    int cell_idx = thrust::get<3>(t);
    // 返回是否为 halo
    return cells[cell_idx]._is_halo;
  }
};

void LinkedCell::ClearDataHalo() {
  if (_ghost_atoms_num <= 0) {
    return;
  }
  _ghost_atoms_num = 0;
  _total_atoms_num = _native_atoms_num;
  ResizePerAtomVectors(_total_atoms_num);
}

void LinkedCell::ResizeDataHalo(size_t halo_size) {
  if (halo_size == 0) {
    return;
  }
  const rbmd::Id target_total =
      _native_atoms_num + _ghost_atoms_num + static_cast<rbmd::Id>(halo_size);
  ResizePerAtomVectors(target_total);
}

void LinkedCell::SetHaloCutoff(rbmd::Real halo_cutoff) {
  rbmd::neighbor::ValidateHaloCutoffCoversNeighborCutoff(
      halo_cutoff, _neighbor_cutoff, "LinkedCell::SetHaloCutoff");
  _halo_cutoff = halo_cutoff;
}

void LinkedCell::SetCoversWholeDomain(int dim, bool covers) {
  if (dim < 0 || dim >= 3) {
    return;
  }
  _covers_whole_domain[dim] = covers;
  if (_linked_cell_device_data_ptr != nullptr) {
    CHECK_RUNTIME(MEMCPY(_linked_cell_device_data_ptr->_d_covers_whole_domain,
                         this->_covers_whole_domain, ALIGN_SIZE(bool, 3), H2D));
  }
}

void LinkedCell::ResizePerAtomVectors(rbmd::Id target_total_atoms) {
  const rbmd::Id safe_total = std::max<rbmd::Id>(target_total_atoms, 0);
  const rbmd::Id safe_total_virial = 6 * safe_total;

  ResizeVector(_per_atom_cell_id, safe_total);
  ResizeVector(_cell_sorted_cell_id, safe_total);
  ResizeVector(_cell_sorted_atom_indices, safe_total);
  ResizeVector(_device_data->_d_unwarp_px, safe_total);
  ResizeVector(_device_data->_d_unwarp_py, safe_total);
  ResizeVector(_device_data->_d_unwarp_pz, safe_total);
  ResizeVector(_device_data->_d_shake_px, safe_total);
  ResizeVector(_device_data->_d_shake_py, safe_total);
  ResizeVector(_device_data->_d_shake_pz, safe_total);
  ResizeVector(_device_data->_d_atoms_id, safe_total);
  ResizeVector(_device_data->_d_atoms_type, safe_total);
  ResizeVector(_device_data->_d_molecular_id, safe_total);
  ResizeVector(_device_data->_d_flagX, safe_total);
  ResizeVector(_device_data->_d_flagY, safe_total);
  ResizeVector(_device_data->_d_flagZ, safe_total);
  ResizeVector(_device_data->_d_px, safe_total);
  ResizeVector(_device_data->_d_py, safe_total);
  ResizeVector(_device_data->_d_pz, safe_total);
  ResizeVector(_device_data->_d_vx, safe_total);
  ResizeVector(_device_data->_d_vy, safe_total);
  ResizeVector(_device_data->_d_vz, safe_total);
  ResizeVector(_device_data->_d_shake_vx, safe_total);
  ResizeVector(_device_data->_d_shake_vy, safe_total);
  ResizeVector(_device_data->_d_shake_vz, safe_total);
  ResizeVector(_device_data->_d_shake_dx, safe_total);
  ResizeVector(_device_data->_d_shake_dy, safe_total);
  ResizeVector(_device_data->_d_shake_dz, safe_total);
  ResizeVector(_device_data->_d_shake_dvx, safe_total);
  ResizeVector(_device_data->_d_shake_dvy, safe_total);
  ResizeVector(_device_data->_d_shake_dvz, safe_total);
  ResizeVector(_device_data->_d_charge, safe_total);
  ResizeVector(_device_data->_d_fx, safe_total);
  ResizeVector(_device_data->_d_fy, safe_total);
  ResizeVector(_device_data->_d_fz, safe_total);
  ResizeVector(_device_data->_d_force_ljcoul_x, safe_total);
  ResizeVector(_device_data->_d_force_ljcoul_y, safe_total);
  ResizeVector(_device_data->_d_force_ljcoul_z, safe_total);
  ResizeVector(_device_data->_d_force_specialcoul_x, safe_total);
  ResizeVector(_device_data->_d_force_specialcoul_y, safe_total);
  ResizeVector(_device_data->_d_force_specialcoul_z, safe_total);
  ResizeVector(_device_data->_d_force_kspace_x, safe_total);
  ResizeVector(_device_data->_d_force_kspace_y, safe_total);
  ResizeVector(_device_data->_d_force_kspace_z, safe_total);
  ResizeVector(_device_data->_d_force_bond_x, safe_total);
  ResizeVector(_device_data->_d_force_bond_y, safe_total);
  ResizeVector(_device_data->_d_force_bond_z, safe_total);
  ResizeVector(_device_data->_d_force_angle_x, safe_total);
  ResizeVector(_device_data->_d_force_angle_y, safe_total);
  ResizeVector(_device_data->_d_force_angle_z, safe_total);
  ResizeVector(_device_data->_d_force_dihedral_x, safe_total);
  ResizeVector(_device_data->_d_force_dihedral_y, safe_total);
  ResizeVector(_device_data->_d_force_dihedral_z, safe_total);
  ResizeVector(_device_data->_d_force_improper_x, safe_total);
  ResizeVector(_device_data->_d_force_improper_y, safe_total);
  ResizeVector(_device_data->_d_force_improper_z, safe_total);
  ResizeVector(_device_data->_d_f2_x, safe_total);
  ResizeVector(_device_data->_d_f2_y, safe_total);
  ResizeVector(_device_data->_d_f2_z, safe_total);
  ResizeVector(_device_data->_d_f3_x, safe_total);
  ResizeVector(_device_data->_d_f3_y, safe_total);
  ResizeVector(_device_data->_d_f3_z, safe_total);
  ResizeVector(_device_data->_d_flat_virial, safe_total_virial);
  ResizeVector(_device_data->_d_flat_virial_lj, safe_total_virial);
  ResizeVector(_device_data->_d_flat_virial_specialcoul, safe_total_virial);
  ResizeVector(_device_data->_d_flat_virial_kspace, safe_total_virial);
  ResizeVector(_device_data->_d_flat_virial_bond_atom, safe_total_virial);
  ResizeVector(_device_data->_d_flat_virial_angle_atom, safe_total_virial);
  ResizeVector(_device_data->_d_flat_virial_dihedral_atom, safe_total_virial);
  ResizeVector(_device_data->_d_flat_virial_improper_atom, safe_total_virial);

  if (_device_data->nmax < safe_total) {
    _device_data->ResizeTopology(safe_total);
  }
}
