#include "domain_decomposition/domain_decomposition.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thrust/count.h>
#include <unistd.h>
#include <vector>

#include <cstddef>

#include "common/startup_phase_debug.h"
#include "common/neighbor_skin.h"
#include "common/rbmd_define.h"
#include "data_manager.h"
#include "domain_decomposition_op.h"
#include "full_shell.h"
#include "halo_leaving_atoms.h"
#include "indirect_neighbour_communication_scheme.h"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"
#include "rbmd_parallel_until_locator.h"
#include "topology_cutoff_calculator.h"

DomainDecomposition::DomainDecomposition(int current_rank, int total_ranks) {
  rbmd::debug::StartupPhaseLog("domain_decomposition.ctor.begin");
  this->h_box = DataManager::getInstance().getMDData()->_box.get();
  //这个盒子是空的，全局放在global info里面了
  this->_mpi_comm = MPI_COMM_WORLD;
  this->_grid_size = {0, 0, 0};
  this->_current_rank = current_rank;
  this->_total_ranks = total_ranks;
  const auto pair_cutoff =
      DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
          "cut_off", "hyper_parameters", "neighbor");
  const auto neighbor_skin = rbmd::neighbor::ReadSkinOrDefault(
      DataManager::getInstance().getConfigData().get());
  const auto neighbor_cutoff = rbmd::neighbor::NeighborCutoff(
      pair_cutoff, neighbor_skin, "DomainDecomposition");
  const auto topology_cutoff =
      TopologyCutoffCalculator::Calculate(neighbor_skin);
  const auto halo_cutoff = MAX(neighbor_cutoff, topology_cutoff);
  rbmd::neighbor::ValidateHaloCutoffCoversNeighborCutoff(
      halo_cutoff, neighbor_cutoff, "DomainDecomposition");

  InitMPIGridDims(halo_cutoff);
  rbmd::debug::StartupPhaseLog("domain_decomposition.ctor.after_init_mpi_grid_dims");
  GetSubDomainBox(h_box);
  rbmd::debug::StartupPhaseLog("domain_decomposition.ctor.after_get_subdomain_box");
  this->_device_data = DataManager::getInstance().getDeviceData();

  LinkedCellLocator::GetInstance().SetHaloCutoff(halo_cutoff);

  this->SetCommunicationScheme("indirect", "fs");
  rbmd::debug::StartupPhaseLog("domain_decomposition.ctor.after_set_communication_scheme");
  this->InitCommunicationPartners(halo_cutoff);
  rbmd::debug::StartupPhaseLog("domain_decomposition.ctor.after_init_communication_partners");
}

void DomainDecomposition::GetSubDomainBox(Box *h_box) {
  this->_h_global_box = std::make_shared<Box>(
      GET_RBMD_PARALLEL->_global_structure_info.global_box);
  for (int dimension = 0; dimension < 3; ++dimension) {
    h_box->_coord_min[dimension] = _h_global_box->_coord_min[dimension] +
                                   _coords[dimension] *
                                       _h_global_box->_length[dimension] /
                                       _grid_size[dimension];
    if (_coords[dimension] + 1 == _grid_size[dimension]) {
      h_box->_coord_max[dimension] = _h_global_box->_coord_min[dimension] +
                                     _h_global_box->_length[dimension];
    } else {
      h_box->_coord_max[dimension] = _h_global_box->_coord_min[dimension] +
                                     (_coords[dimension] + 1) *
                                         _h_global_box->_length[dimension] /
                                         _grid_size[dimension];
    }
  }
  bool pbc[3] = {_h_global_box->_pbc_x, _h_global_box->_pbc_y,
                 _h_global_box->_pbc_z};
  h_box->Setup(h_box->_type, h_box->_coord_min, h_box->_coord_max, pbc);
  int current_rank = 100;
  MPI_Comm_rank(_mpi_comm, &current_rank);

  // if (current_rank == 0) {
  //   std::cout<< "RANK  " << current_rank << "    " <<
  //   "================================================GLOBAL
  //   BOX==========================================================================="
  //   <<std::endl; std::cout << "Min: (" << _h_global_box->_coord_min[0] << ",
  //   " << _h_global_box->_coord_min[1] << ", " << _h_global_box->_coord_min[2]
  //   << ")\n"
  //          << "Max: (" << _h_global_box->_coord_max[0] << ", " <<
  //          _h_global_box->_coord_max[1] << ", " <<
  //          _h_global_box->_coord_max[2] << ")\n"
  //          << "Size: ("
  //          << _h_global_box->_length[0] << ", "
  //          << _h_global_box->_length[1]  << ", "
  //          << _h_global_box->_length[2]  << ")\n";
  //   std::cout<< "RANK  " << current_rank << "    " <<
  //   "================================================LOCAL
  //   BOX==========================================================================="
  //   <<std::endl;
  //
  //   std::cout << "Min: (" << h_box->_coord_min[0] << ", " <<
  //   h_box->_coord_min[1] << ", " << h_box->_coord_min[2] << ")\n"
  //      << "Max: (" << h_box->_coord_max[0] << ", " << h_box->_coord_max[1] <<
  //      ", " << h_box->_coord_max[2] << ")\n"
  //      << "Size: ("
  //      << h_box->_length[0] << ", "
  //      << h_box->_length[1]  << ", "
  //      << h_box->_length[2]  << ")\n";
  // }
  // OutputLocalBoxInfoToFile();
  // MPI_Barrier(MPI_COMM_WORLD);
}

std::vector<CommunicationPartner>
DomainDecomposition::GetNeighboursFromHaloRegion(const HaloRegion &haloRegion) {
  // TODO: change this method for support of midpoint rule, half shell, eighth
  // shell, Neutral Territory
  //  currently only one process per region is possible.
  int rank;
  int regionCoords[3];
  for (unsigned int d = 0; d < 3; d++) {
    regionCoords[d] = _coords[d] + haloRegion.offset[d];
  }
  // TODO: only full shell! (otherwise more neighbours possible)
  MPI_CHECK(MPI_Cart_rank(_mpi_comm, regionCoords, &rank));
  // does automatic shift for periodic boundaries   就是那个PBC自动找邻居线程
  // 非点对点不需要mpicomm
  rbmd::Real haloLow[3];
  rbmd::Real haloHigh[3];
  rbmd::Real boundaryLow[3];
  rbmd::Real boundaryHigh[3];
  rbmd::Real shift[3];
  bool enlarged[3][2];

  for (unsigned int d = 0; d < 3; d++) {
    haloLow[d] = haloRegion.rmin[d];
    haloHigh[d] = haloRegion.rmax[d];
    // TODO: ONLY FULL SHELL!!! 啊？

    boundaryLow[d] =
        haloRegion.rmin[d] -
        haloRegion.offset[d] * haloRegion.width; // rmin[d] if offset[d]==0
    boundaryHigh[d] =
        haloRegion.rmax[d] - haloRegion.offset[d] * haloRegion.width;
    // if offset[d]!=0 : shift by cutoff in negative offset direction
    if (_coords[d] == 0 and haloRegion.offset[d] == -1) {
      shift[d] = _h_global_box->_length[d];
    } else if (_coords[d] == _grid_size[d] - 1 and haloRegion.offset[d] == 1) {
      shift[d] = -_h_global_box->_length[d];
    } else {
      shift[d] = 0.;
    }
    enlarged[d][0] = false;
    enlarged[d][1] = false;
  }
  // initialize using initializer list - here a vector with one element is
  // created
  std::vector<CommunicationPartner> temp;
  temp.emplace_back(rank, haloLow, haloHigh, boundaryLow, boundaryHigh, shift,
                    haloRegion.offset, enlarged); // TODO  减少拷贝
  return temp;
}

void DomainDecomposition::InitCommunicationPartners(const rbmd::Real cutoff) {
  rbmd::debug::StartupPhaseLog(
      "domain_decomposition.init_communication_partners.begin",
      "cutoff=" + std::to_string(static_cast<double>(cutoff)));
  auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  for (int d = 0; d < 3; ++d) {
    _neighbour_coneighbour_communication_scheme->setCoverWholeDomain(
        d, _grid_size[d] == 1);
    linked_cell->SetCoversWholeDomain(d, _grid_size[d] == 1);
  }
  _neighbour_coneighbour_communication_scheme->initCommunicationPartners(cutoff,
                                                                         this);
  rbmd::debug::StartupPhaseLog(
      "domain_decomposition.init_communication_partners.end");
}

void DomainDecomposition::SetCommunicationScheme(
    const std::string &comm_scheme, const std::string &zonal_method) {
  // TODO enum auto
  ZonalMethod *_zonal_method = nullptr;
  if (zonal_method == "fs") {
    _zonal_method = new FullShell();
  } else {
    throw std::runtime_error("暂未实现的方法");
  }

  if (comm_scheme == "indirect") {
    this->_neighbour_coneighbour_communication_scheme =
        std::make_shared<IndirectNeighbourCommunicationScheme>(_zonal_method);
  }
}

// 只是为了获取周期性边界条件是否使用，暂时这样，与盒子类型实际上是无关的。
void DomainDecomposition::InitMPIGridDims(rbmd::Real interaction_length) {
  rbmd::debug::StartupPhaseLog("domain_decomposition.init_mpi_grid_dims.begin");
  auto h_box = GET_RBMD_PARALLEL->_global_structure_info.global_box;
  int period[3] = {h_box._pbc_x, h_box._pbc_y, h_box._pbc_z};
  // TODO 暂时弄不太明白  目前来看应该是进程查找的PBC应用
  // 进程上面没进程就找到下面那个进程
  int reorder = 0;
  {
    auto num_procs_grid_size =
        _grid_size[0] * _grid_size[1] * _grid_size[2]; // TODO What doing?
    if (num_procs_grid_size != _total_ranks and num_procs_grid_size != 0) {
      throw std::runtime_error("error mpi grdims!"); // TODO log
    }
  }
  MPI_CHECK(MPI_Dims_create(_total_ranks, 3, _grid_size.data()));
  rbmd::debug::StartupPhaseLog(
      "domain_decomposition.init_mpi_grid_dims.after_dims_create",
      "dims=" + std::to_string(_grid_size[0]) + "x" +
          std::to_string(_grid_size[1]) + "x" + std::to_string(_grid_size[2]));
  ValidateResolvableGrid(interaction_length);
  MPI_CHECK(MPI_Cart_create(_mpi_comm, 3, _grid_size.data(), period, reorder,
                            &_mpi_comm));
  rbmd::debug::StartupPhaseLog(
      "domain_decomposition.init_mpi_grid_dims.after_cart_create");
  std::cout << "MPI grid dimensions: " << _grid_size[0] << ", " << _grid_size[1]
            << ", " << _grid_size[2] << std::endl;
  MPI_CHECK(MPI_Comm_rank(_mpi_comm, &_current_rank));
  MPI_CHECK(MPI_Cart_coords(_mpi_comm, _current_rank, 3, _coords));
  rbmd::debug::StartupPhaseLog(
      "domain_decomposition.init_mpi_grid_dims.after_cart_coords",
      "coords=" + std::to_string(_coords[0]) + "," +
          std::to_string(_coords[1]) + "," + std::to_string(_coords[2]));
  std::cout << "MPI coordinate of current process: " << _coords[0] << ", "
            << _coords[1] << ", " << _coords[2] << std::endl;
}

void DomainDecomposition::ValidateResolvableGrid(
    rbmd::Real interaction_length) const {
  constexpr int kMinCellsPerRankPerDimension = 2;

  if (_total_ranks <= 1) {
    return;
  }

  if (!std::isfinite(static_cast<double>(interaction_length)) ||
      interaction_length <= rbmd::Real(0)) {
    std::ostringstream oss;
    oss << "DomainDecomposition invalid interaction_length="
        << interaction_length;
    throw std::runtime_error(oss.str());
  }

  const auto config = DataManager::getInstance().getConfigData();
  const auto neighbor_skin = rbmd::neighbor::ReadSkinOrDefault(config.get());
  const auto pair_cutoff =
      config->Get<rbmd::Real>("cut_off", "hyper_parameters", "neighbor");
  if (!std::isfinite(static_cast<double>(pair_cutoff)) ||
      pair_cutoff <= rbmd::Real(0)) {
    std::ostringstream oss;
    oss << "DomainDecomposition invalid neighbor cut_off=" << pair_cutoff;
    throw std::runtime_error(oss.str());
  }
  const auto neighbor_cutoff = rbmd::neighbor::NeighborCutoff(
      pair_cutoff, neighbor_skin, "DomainDecomposition");

  rbmd::Id cells_in_cutoff_radius = 1;
  if (config->Get<std::string>("type", "hyper_parameters", "neighbor") ==
      "RBL") {
    const auto r_core =
        config->Get<rbmd::Real>("r_core", "hyper_parameters", "neighbor");
    if (!std::isfinite(static_cast<double>(r_core)) ||
        r_core <= rbmd::Real(0) || r_core >= pair_cutoff) {
      std::ostringstream oss;
      oss << "DomainDecomposition invalid RBL r_core=" << r_core
          << ", cut_off=" << pair_cutoff;
      throw std::runtime_error(oss.str());
    }
    cells_in_cutoff_radius = static_cast<rbmd::Id>(
        std::ceil(static_cast<double>(neighbor_cutoff / r_core)));
  }

  const rbmd::Real cell_length =
      neighbor_cutoff / static_cast<rbmd::Real>(cells_in_cutoff_radius);
  const auto& global_box = GET_RBMD_PARALLEL->_global_structure_info.global_box;
  rbmd::Id global_cells_per_dim[3]{};
  rbmd::Id required_cells[3]{};
  rbmd::Real subdomain_length[3]{};

  for (int dim = 0; dim < 3; ++dim) {
    const auto global_length = global_box._length[dim];
    if (!std::isfinite(static_cast<double>(global_length)) ||
        global_length <= rbmd::Real(0)) {
      std::ostringstream oss;
      oss << "DomainDecomposition invalid global box length at dim " << dim
          << ": " << global_length;
      throw std::runtime_error(oss.str());
    }
    if (_grid_size[dim] <= 0) {
      std::ostringstream oss;
      oss << "DomainDecomposition invalid MPI grid size at dim " << dim
          << ": " << _grid_size[dim];
      throw std::runtime_error(oss.str());
    }

    subdomain_length[dim] =
        global_length / static_cast<rbmd::Real>(_grid_size[dim]);
    global_cells_per_dim[dim] = static_cast<rbmd::Id>(
        std::floor(static_cast<double>(global_length / cell_length)));
    required_cells[dim] =
        static_cast<rbmd::Id>(_grid_size[dim] *
                              kMinCellsPerRankPerDimension);
    if (subdomain_length[dim] < interaction_length ||
        global_cells_per_dim[dim] < required_cells[dim]) {
      std::ostringstream oss;
      oss << "DomainDecomposition grid is not resolvable at dim " << dim
          << ": global_length=" << global_length
          << ", subdomain_length=" << subdomain_length[dim]
          << ", interaction_length=" << interaction_length
          << ", cut_off=" << pair_cutoff
          << ", skin=" << neighbor_skin
          << ", neighbor_cutoff=" << neighbor_cutoff
          << ", cells_in_cutoff_radius=" << cells_in_cutoff_radius
          << ", global_cells_per_dim=" << global_cells_per_dim[dim]
          << ", required_cells=" << required_cells[dim]
          << ", mpi_grid=(" << _grid_size[0] << ", " << _grid_size[1]
          << ", " << _grid_size[2] << ")"
          << ". Use fewer MPI ranks, increase the box size, or reduce the "
             "interaction cutoff.";
      throw std::runtime_error(oss.str());
    }
  }
}

// 参考的代码这里到这里是把halo都删除了，实际上可能不删除。在设备上反复读写很正常，但是halo的数量会变化。
// 可以halo另外维护一个数组，超过本地的索引就去这个数组找，这样就能大量减少内存操作了。
//! 然后刚开始会通信halo的数量的。
// TODO Deleate!
void DomainDecomposition::HandleDomainLeavingAtoms(
    unsigned dim, LinkedCell *linked_cell) const {
  rbmd::Real shiftMagnitude = h_box->_coord_max[dim] - h_box->_coord_min[dim];
  const int sDim = dim + 1;
  // 这个是动态的，不好放入核函数
  rbmd::Real *d_start_region;
  rbmd::Real *d_end_region;
  MALLOC(&d_start_region, ALIGN_SIZE(rbmd::Real, 3));
  MALLOC(&d_end_region, ALIGN_SIZE(rbmd::Real, 3));
  for (int direction = -sDim; direction < 2 * sDim; direction += 2 * sDim) {
    rbmd::Real shift = copysign(shiftMagnitude, -direction);
    rbmd::Real cutoff = linked_cell->_halo_cutoff;
    rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) start_region[3] = {
        h_box->_coord_min[0] - cutoff, h_box->_coord_min[1] - cutoff,
        h_box->_coord_min[2] - cutoff};
    rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) end_region[3] = {
        h_box->_coord_max[0] + cutoff, h_box->_coord_max[1] + cutoff,
        h_box->_coord_max[2] + cutoff}; // 因为halo删除了，所以只要判断这个范围
    if (direction < 0) {
      end_region[dim] = h_box->_coord_min[dim];
    } else {
      start_region[dim] = h_box->_coord_max[dim];
    }
    CHECK_RUNTIME(
        MEMCPY(d_start_region, start_region, ALIGN_SIZE(rbmd::Real, 3), H2D));
    CHECK_RUNTIME(
        MEMCPY(d_end_region, end_region, ALIGN_SIZE(rbmd::Real, 3), H2D));
    // do not need clamp iterated region to local MPI subdomain
    rbmd::Real *mod_pos = nullptr;
    if (dim == 0)
      mod_pos = thrust::raw_pointer_cast(_device_data->_d_px.data());
    if (dim == 1)
      mod_pos = thrust::raw_pointer_cast(_device_data->_d_py.data());
    if (dim == 2)
      mod_pos = thrust::raw_pointer_cast(_device_data->_d_pz.data());
    op::HandleDomainLeavingAtomsOneDimOp<device::DEVICE_GPU>
        handle_domain_leaving_atoms_one_dim_op;
    handle_domain_leaving_atoms_one_dim_op(
        d_start_region, d_end_region,
        thrust::raw_pointer_cast(_device_data->_d_px.data()),
        thrust::raw_pointer_cast(_device_data->_d_py.data()),
        thrust::raw_pointer_cast(_device_data->_d_pz.data()), mod_pos, *h_box,
        dim, shift, linked_cell->_native_atoms_num);
  }
  CHECK_RUNTIME(FREE(d_start_region));
  CHECK_RUNTIME(FREE(d_end_region));
}

rbmd::Id GetDimHaloAtomCountOLD(int dim, const rbmd::Id _cellsPerDimension[3],
                                const LinkedCell *linked_cell) {
  thrust::device_vector<int> d_atoms_count(linked_cell->_cells.size());

  thrust::transform(linked_cell->_cells.begin(), linked_cell->_cells.end(),
                    d_atoms_count.begin(), [] __device__(const Cell &cell) {
                      return cell._atoms_count;
                    });

  thrust::device_vector<int> h_atoms_count = d_atoms_count;

  rbmd::Id current_cell_idx = 0;
  rbmd::Id total_halo_atom_count = 0;

  // 确保dim在合法范围内
  if (dim < 0 || dim > 2) {
    throw std::invalid_argument("Invalid dimension specified.");
  }

  if (_cellsPerDimension[dim] <=
      2 * linked_cell->_cell_count_within_cutoff + 1) {
    throw std::runtime_error("Halo Ovleap");
  }

  for (int iz = 0; iz < _cellsPerDimension[2]; ++iz) {
    for (int iy = 0; iy < _cellsPerDimension[1]; ++iy) {
      for (int ix = 0; ix < _cellsPerDimension[0]; ++ix) {
        // 检查当前cell是否在指定维度的halo区域内
        bool is_halo = false;

        if (dim == 0) {
          // x维度的halo: 从边界向内_cell_count_within_cutoff层
          is_halo = (ix >= 1 && ix <= linked_cell->_cell_count_within_cutoff) ||
                    (ix >= _cellsPerDimension[0] - 1 -
                               linked_cell->_cell_count_within_cutoff &&
                     ix < _cellsPerDimension[0] - 1);
        } else if (dim == 1) {
          // y维度的halo
          is_halo = (iy >= 1 && iy <= linked_cell->_cell_count_within_cutoff) ||
                    (iy >= _cellsPerDimension[1] - 1 -
                               linked_cell->_cell_count_within_cutoff &&
                     iy < _cellsPerDimension[1] - 1);
        } else if (dim == 2) {
          // z维度的halo
          is_halo = (iz >= 1 && iz <= linked_cell->_cell_count_within_cutoff) ||
                    (iz >= _cellsPerDimension[2] - 1 -
                               linked_cell->_cell_count_within_cutoff &&
                     iz < _cellsPerDimension[2] - 1);
        }

        if (is_halo) {
          current_cell_idx =
              (iz * _cellsPerDimension[1] + iy) * _cellsPerDimension[0] + ix;
          total_halo_atom_count += h_atoms_count[current_cell_idx];
        }
      }
    }
  }

  return total_halo_atom_count;
}

rbmd::Id GetDimHaloAtomCount(
    int dim,
    const LinkedCell *linked_cell, // 用于获取 cutoff 和 native_atoms_num
    const Box &h_box, // 主机端的盒子边界信息 (您代码中的 h_box)
    // 假设您的 _device_data 包含指向设备粒子坐标的指针
    // 如果不是直接的成员，您可能需要调整如何传递这些指针
    const rbmd::Real *d_native_px, // 指向设备上本地粒子x坐标的指针
    const rbmd::Real *d_native_py, // 指向设备上本地粒子y坐标的指针
    const rbmd::Real *d_native_pz // 指向设备上本地粒子z坐标的指针
) {
  rbmd::Id total_halo_atom_count = 0;
  rbmd::Real cutoff = linked_cell->_halo_cutoff;
  rbmd::Id native_atoms_num = linked_cell->_native_atoms_num;

  if (native_atoms_num == 0) {
    return 0;
  }

  if (dim < 0 || dim > 2) {
    throw std::invalid_argument(
        "Invalid dimension specified in GetDimHaloAtomCount.");
  }

  // 为计数核函数准备设备内存
  rbmd::Real *d_temp_start_region;
  rbmd::Real *d_temp_end_region;
  unsigned int *d_single_direction_counter; // 用于存储单向计数的设备内存

  CHECK_RUNTIME(MALLOC(&d_temp_start_region, ALIGN_SIZE(rbmd::Real, 3)));
  CHECK_RUNTIME(MALLOC(&d_temp_end_region, ALIGN_SIZE(rbmd::Real, 3)));
  CHECK_RUNTIME(MALLOC(&d_single_direction_counter, sizeof(unsigned int)));

  const int sDim = dim + 1; // 与您的 PopulateHaloLayerWithCopies 逻辑一致

  // 这个循环与您 PopulateHaloLayerWithCopies 中的主循环一致
  for (int direction_loop_var = -sDim; direction_loop_var < 2 * sDim;
       direction_loop_var += 2 * sDim) {
    rbmd::Real h_start_region_slab[3]; // 主机端临时存储区域边界
    rbmd::Real h_end_region_slab[3];   // 主机端临时存储区域边界

    // 初始化搜索区域（与 PopulateHaloLayerWithCopies 中一致）
    // 这些边界定义了源粒子可能存在的区域
    h_start_region_slab[0] = h_box._coord_min[0] - cutoff;
    h_start_region_slab[1] = h_box._coord_min[1] - cutoff;
    h_start_region_slab[2] = h_box._coord_min[2] - cutoff;
    h_end_region_slab[0] = h_box._coord_max[0] + cutoff;
    h_end_region_slab[1] = h_box._coord_max[1] + cutoff;
    h_end_region_slab[2] = h_box._coord_max[2] + cutoff;

    // 根据当前方向调整指定维度 dim 的区域边界
    if (direction_loop_var < 0) {
      // 对应 PopulateHaloLayerWithCopies 中 direction < 0 的情况
      h_start_region_slab[dim] = h_box._coord_min[dim];
      h_end_region_slab[dim] = h_box._coord_min[dim] + cutoff;
    } else {
      h_start_region_slab[dim] = h_box._coord_max[dim] - cutoff;
      h_end_region_slab[dim] = h_box._coord_max[dim];
    }

    // 将区域边界从主机复制到设备
    CHECK_RUNTIME(MEMCPY(d_temp_start_region, h_start_region_slab,
                         ALIGN_SIZE(rbmd::Real, 3), H2D));
    CHECK_RUNTIME(MEMCPY(d_temp_end_region, h_end_region_slab,
                         ALIGN_SIZE(rbmd::Real, 3), H2D));

    // 重置设备上的单向计数器为0
    unsigned int zero = 0;
    CHECK_RUNTIME(
        MEMCPY(d_single_direction_counter, &zero, sizeof(unsigned int), H2D));

    op::CountHaloCandidatesOp<device::DEVICE_GPU> count_halo_candidates_op;
    count_halo_candidates_op(d_temp_start_region, d_temp_end_region,
                             d_native_px, d_native_py, d_native_pz,
                             native_atoms_num, d_single_direction_counter);
    // 检查CUDA错误并在必要时同步是个好习惯
    // مثلاً: cudaError_t err = cudaGetLastError(); if (err != cudaSuccess)
    // printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // cudaDeviceSynchronize(); // 确保计数完成

    // 将计数结果从设备复制回主机
    unsigned int count_for_this_direction;
    CHECK_RUNTIME(MEMCPY(&count_for_this_direction, d_single_direction_counter,
                         sizeof(unsigned int), D2H));
    total_halo_atom_count += count_for_this_direction;
  }

  // 释放为计数分配的临时设备内存
  CHECK_RUNTIME(FREE(d_temp_start_region));
  CHECK_RUNTIME(FREE(d_temp_end_region));
  CHECK_RUNTIME(FREE(d_single_direction_counter));

  return total_halo_atom_count;
}

void DomainDecomposition::PopulateHaloLayerWithCopies(
    unsigned dim, LinkedCell *linked_cell) const {
  // 调用新的计数函数
  auto halo_size = GetDimHaloAtomCount(
      dim, linked_cell,
      *h_box, // 传递主机盒子信息
      thrust::raw_pointer_cast(
          _device_data->_d_px.data()), //传递设备粒子数据指针
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()));

  std::cout << "Halo Size :  " << halo_size << std::endl;
  unsigned int pre_ghots_num = linked_cell->_ghost_atoms_num; // 第一次进来是0
  linked_cell->ResizeDataHalo(halo_size);
  linked_cell->UpdateGhostNum(static_cast<rbmd::Id>(halo_size));
  rbmd::Real shiftMagnitude = h_box->_coord_max[dim] - h_box->_coord_min[dim];
  const int sDim = dim + 1;

  // 这个是动态的，不好放入核函数
  rbmd::Real *d_start_region;
  rbmd::Real *d_end_region;
  unsigned int *d_counter; // 添加计数器

  CHECK_RUNTIME(MALLOC(&d_start_region, ALIGN_SIZE(rbmd::Real, 3)));
  CHECK_RUNTIME(MALLOC(&d_end_region, ALIGN_SIZE(rbmd::Real, 3)));
  CHECK_RUNTIME(MALLOC(&d_counter, sizeof(unsigned int)));
  // 重置计数器
  unsigned int zero = 0;
  CHECK_RUNTIME(MEMCPY(d_counter, &zero, sizeof(unsigned int), H2D));
  for (int direction = -sDim; direction < 2 * sDim; direction += 2 * sDim) {
    rbmd::Real shift = copysign(shiftMagnitude, -direction);
    rbmd::Real cutoff = linked_cell->_halo_cutoff;
    rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) start_region[3] = {
        h_box->_coord_min[0] - cutoff, h_box->_coord_min[1] - cutoff,
        h_box->_coord_min[2] - cutoff};
    rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) end_region[3] = {
        h_box->_coord_max[0] + cutoff, h_box->_coord_max[1] + cutoff,
        h_box->_coord_max[2] + cutoff}; // 因为halo删除了，所以只要判断这个范围

    if (direction < 0) {
      start_region[dim] = h_box->_coord_min[dim];
      end_region[dim] = h_box->_coord_min[dim] + cutoff;
    } else {
      start_region[dim] = h_box->_coord_max[dim] - cutoff;
      end_region[dim] = h_box->_coord_max[dim];
    }
    CHECK_RUNTIME(
        MEMCPY(d_start_region, start_region, ALIGN_SIZE(rbmd::Real, 3), H2D));
    CHECK_RUNTIME(
        MEMCPY(d_end_region, end_region, ALIGN_SIZE(rbmd::Real, 3), H2D));

    op::PopulateHaloOp<device::DEVICE_GPU> populate_halo_op;
    populate_halo_op( // 修正指针问题
        d_start_region, d_end_region,
        thrust::raw_pointer_cast(_device_data->_d_px.data()),
        thrust::raw_pointer_cast(_device_data->_d_py.data()),
        thrust::raw_pointer_cast(_device_data->_d_pz.data()), *h_box,
        thrust::raw_pointer_cast(_device_data->_d_px.data()) +
            linked_cell->_native_atoms_num + pre_ghots_num,
        thrust::raw_pointer_cast(_device_data->_d_py.data()) +
            linked_cell->_native_atoms_num + pre_ghots_num,
        thrust::raw_pointer_cast(_device_data->_d_pz.data()) +
            linked_cell->_native_atoms_num + pre_ghots_num,
        d_counter, // 传递计数器到核函数
        halo_size, dim, shift,
        linked_cell->_native_atoms_num, // 改为native_atoms_num而不是total
        // 拓扑相关参数：传递 atom_id 和 atom_type 用于 bond/angle 力计算
        thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
        thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
        thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()) +
            linked_cell->_native_atoms_num + pre_ghots_num,
        thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()) +
            linked_cell->_native_atoms_num + pre_ghots_num);
  }

  CHECK_RUNTIME(FREE(d_start_region));
  CHECK_RUNTIME(FREE(d_end_region));
  CHECK_RUNTIME(FREE(d_counter));
}

void DomainDecomposition::ExchangeMoleculesMPI(LinkedCell *moleculeContainer) {
  this->_neighbour_coneighbour_communication_scheme->exchangeMoleculesMPI(
      moleculeContainer, this);
}

void DomainDecomposition::OutputLocalBoxInfoToFile() {
  // 创建数据结构来存储盒子信息
  struct BoxInfo {
    int rank;
    int grid_coords[3]; // MPI网格中的坐标
    double local_min[3];
    double local_max[3];
  };

  BoxInfo local_info;
  local_info.rank = this->_current_rank;
  local_info.grid_coords[0] = this->_coords[0];
  local_info.grid_coords[1] = this->_coords[1];
  local_info.grid_coords[2] = this->_coords[2];

  // 填充local盒子信息
  local_info.local_min[0] = this->h_box->_coord_min[0];
  local_info.local_min[1] = this->h_box->_coord_min[1];
  local_info.local_min[2] = this->h_box->_coord_min[2];
  local_info.local_max[0] = this->h_box->_coord_max[0];
  local_info.local_max[1] = this->h_box->_coord_max[1];
  local_info.local_max[2] = this->h_box->_coord_max[2];

  // 收集所有进程的信息到rank 0
  std::vector<BoxInfo> all_box_info;
  if (this->_current_rank == 0) {
    all_box_info.resize(this->_total_ranks);
  }

  // 使用MPI_Gather收集所有数据到rank 0
  MPI_Gather(&local_info, sizeof(BoxInfo), MPI_BYTE, all_box_info.data(),
             sizeof(BoxInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

  // 只有rank 0负责写入文件
  if (this->_current_rank == 0) {
    std::ofstream outfile("localbox_info_all_processes.txt");
    if (outfile.is_open()) {
      outfile << "=================================================\n";
      outfile << "           所有进程LOCAL盒子信息汇总\n";
      outfile << "           总进程数: " << this->_total_ranks << "\n";
      outfile << "           MPI网格维度: [" << this->_grid_size[0] << ", "
              << this->_grid_size[1] << ", " << this->_grid_size[2] << "]\n";
      outfile << "=================================================\n\n";

      // 输出全局盒子信息
      outfile << "全局盒子信息:\n";
      outfile << "  Min: (" << this->_h_global_box->_coord_min[0] << ", "
              << this->_h_global_box->_coord_min[1] << ", "
              << this->_h_global_box->_coord_min[2] << ")\n";
      outfile << "  Max: (" << this->_h_global_box->_coord_max[0] << ", "
              << this->_h_global_box->_coord_max[1] << ", "
              << this->_h_global_box->_coord_max[2] << ")\n";
      outfile << "  Size: (" << this->_h_global_box->_length[0] << ", "
              << this->_h_global_box->_length[1] << ", "
              << this->_h_global_box->_length[2] << ")\n";
      outfile << "=================================================\n\n";

      for (int i = 0; i < this->_total_ranks; ++i) {
        const BoxInfo &info = all_box_info[i];
        outfile << "进程 " << info.rank << " (MPI网格坐标: ["
                << info.grid_coords[0] << ", " << info.grid_coords[1] << ", "
                << info.grid_coords[2] << "] ):\n";
        outfile << "  LOCAL盒子信息:\n";
        outfile << "    X: [" << info.local_min[0] << ", " << info.local_max[0]
                << "]\n";
        outfile << "    Y: [" << info.local_min[1] << ", " << info.local_max[1]
                << "]\n";
        outfile << "    Z: [" << info.local_min[2] << ", " << info.local_max[2]
                << "]\n";

        // 计算local盒子尺寸
        double local_x = info.local_max[0] - info.local_min[0];
        double local_y = info.local_max[1] - info.local_min[1];
        double local_z = info.local_max[2] - info.local_min[2];

        outfile << "  LOCAL盒子尺寸: [" << local_x << ", " << local_y << ", "
                << local_z << "]\n";
        outfile << "------------------------------------------------\n";
      }

      outfile.close();
      std::cout
          << "LOCAL盒子信息已成功写入 localbox_info_all_processes.txt 文件"
          << std::endl;
    } else {
      std::cerr << "无法打开文件 localbox_info_all_processes.txt 进行写入"
                << std::endl;
    }
  }

  // 确保所有进程都完成这个函数后再继续
  MPI_Barrier(MPI_COMM_WORLD);
}
