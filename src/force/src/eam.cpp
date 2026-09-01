#include "eam.h"
#include "force_box_selector.h"

#include <output/include/Logger.hpp>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "../../common/device_types.h"
#include "../../common/rbmd_define.h"
#include "../../common/types.h"
#include "../common/unit_factor.h"
#include "eam_op/eam_op.h"
#include "force_op/force_op.h"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"
#include "neighbor_list/include/neighbor_list_builder/full_neighbor_list_builder.h"
#include "neighbor_list/include/neighbor_list_builder/rbl_full_neighbor_list_builder.h"
#ifdef USE_MPI
#include "rbmd_parallel_until_locator.h"
#endif
#include "thrust/sort.h"
// #include <hipcub/hipcub.hpp>
// #include <hipcub/backend/rocprim/block/block_reduce.hpp>
#include "common/mpi_root_guard.hpp"
#include "common/mpi_reduce_helper.hpp"
#include "common/thermo_stats.hpp"
#include "common/timing_statistics.hpp"
extern rbmd::Id test_current_step;

namespace {

bool ReadDirectTraversal(const std::shared_ptr<ConfigData>& config) {
  return config &&
         config->PathExists(
             {"hyper_parameters", "neighbor", "direct_traversal"}) &&
         config->Get<bool>("direct_traversal", "hyper_parameters", "neighbor");
}

rbmd::Id NeighborCellCount(const LinkedCell& linked_cell) {
  const rbmd::Id width = 2 * linked_cell._cell_count_within_cutoff + 1;
  return width * width * width;
}

void PrepareDirectCellTraversal(const std::shared_ptr<LinkedCell>& linked_cell) {
  if (!linked_cell) {
    throw std::runtime_error("EAM direct traversal requires linked cells");
  }
#ifdef USE_MPI
  linked_cell->ClearDataHalo();
  RbmdParallelUntilLocator::GetInstance()
      .GetRbmdParallelUntil()
      ->BalanceAndExchange();
#endif
  linked_cell->AssignAtomsToCell();
  linked_cell->SortAtomsByCellKey();
  linked_cell->ComputeCellRangesIndices();
}

rbmd::Id ComputeRblSelectionFrequency(const LinkedCell& linked_cell,
                                      const Box& force_box,
                                      rbmd::Real r_core,
                                      rbmd::Real cut_off,
                                      rbmd::Id neighbor_sample_num) {
  const rbmd::Real volume = CalculateVolume(force_box);
  if (!(volume > rbmd::Real(0)) || neighbor_sample_num <= 0) {
    throw std::runtime_error("Invalid EAM direct RBL sampling parameters");
  }
  const rbmd::Real density = linked_cell._total_atoms_num / volume;
  if (!(density > rbmd::Real(0))) {
    throw std::runtime_error("EAM direct RBL requires a positive atom density");
  }

  const rbmd::Real coefficient = 1.0 + (0.05 / density - 0.05);
  const rbmd::Id integer_coefficient =
      static_cast<rbmd::Id>(std::round(static_cast<double>(coefficient)));
  const rbmd::Id core_count =
      integer_coefficient * density *
          std::ceil(4.0 / 3.0 * M_PI * std::pow(r_core, 3)) +
      1;
  const rbmd::Id cutoff_count =
      integer_coefficient * density *
          std::ceil(4.0 / 3.0 * M_PI * std::pow(cut_off, 3)) +
      1;
  const rbmd::Id shell_count = cutoff_count - core_count;
  if (shell_count <= 0) {
    return 1;
  }
  const rbmd::Real random_rate =
      static_cast<rbmd::Real>(neighbor_sample_num) /
      static_cast<rbmd::Real>(shell_count);
  if (random_rate >= rbmd::Real(1)) {
    return 1;
  }
  return MAX(static_cast<rbmd::Id>(1),
             static_cast<rbmd::Id>(std::ceil(1.0 / random_rate)));
}

}  // namespace

EAM::EAM() {
  const auto& config = DataManager::getInstance().getConfigData();
  _direct_traversal = ReadDirectTraversal(config);
  if (!_direct_traversal) {
    _rbl_neighbor_list_builder =
        std::make_shared<RblFullNeighborListBuilder>();
    _neighbor_list_builder = std::make_shared<FullNeighborListBuilder>();
  }

  auto unit = DataManager::getInstance().getConfigData()->Get
  <std::string>("unit", "init_configuration", "read_data");
  UNIT unit_factor = ParseUnit(unit);
  if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    std::remove("thermo.txt");
  }
}

EAM::~EAM() {  }

void EAM::Init() {
  const auto& config = DataManager::getInstance().getConfigData();

  //neighbor
  _cut_off = config->Get<rbmd::Real>("cut_off", "hyper_parameters", "neighbor");
  _neighbor_type = config->Get<std::string>("type", "hyper_parameters", "neighbor");
  if (_direct_traversal && _neighbor_type != "RBL") {
    throw std::runtime_error(
        "hyper_parameters.neighbor.direct_traversal currently requires "
        "force_field.type=EAM and neighbor.type=RBL");
  }
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

  //
  std::string potential_file = config->ResolvePath(
      config->Get<std::string>("potential_file", "hyper_parameters", "force_field"));
  auto& hyper = config->GetJsonNode("hyper_parameters");
  hyper["force_field"]["potential_file"] = potential_file;
  ReadPotentialFile(potential_file);
  InitStyle();
  if (_direct_traversal) {
    Logger::Instance().info(
        "EAM RBL direct linked-cell traversal enabled; neighbor arrays are "
        "not allocated or generated.");
  }
}

void EAM::Execute() {
  SumForces();
  EvaluatePotentialEnergy();
}

void EAM::ReadPotentialFile(const std::string& filename) {
  std::ifstream input_file(filename);
  if (!input_file.is_open())
  {
    Logger::Instance().error("\033[31m Failed to open EAM potential file: {}\033[0m", filename);
    throw std::runtime_error("Failed to open EAM potential file: " + filename);
  }
  for (int i = 0; i < 2; ++i)
  {
    input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  input_file >> file.nrho >> file.drho >> file.nr >> file.dr >> file.cut_off;
  if (input_file.fail() || file.nrho <= 0 || file.nr <= 0) {
    throw std::runtime_error("Invalid EAM potential header in file: " + filename);
  }

  file.frho.resize(file.nrho + 1);
  file.zr.resize(file.nr + 1);
  file.rhor.resize(file.nrho + 1);

  for (int i = 0; i < file.nrho; ++i)
  {
    input_file >> file.frho[i];
    if (input_file.fail()) {
      throw std::runtime_error("Invalid frho table in EAM potential file: " + filename);
    }
  }

  for (int i = 0; i < file.nr; ++i)
  {
    input_file >> file.zr[i];
    if (input_file.fail()) {
      throw std::runtime_error("Invalid zr table in EAM potential file: " + filename);
    }
  }

  for (int i = 0; i < file.nrho; ++i)
  {
    input_file >> file.rhor[i];
    if (input_file.fail()) {
      throw std::runtime_error("Invalid rhor table in EAM potential file: " + filename);
    }
  }

  input_file.close();
}

void EAM::InitStyle() {
  AllocateEAM();
  file2array();
  array2spline();
  SetEAM();
}

void EAM::AllocateEAM() {  }

void EAM::file2array() {
  rbmd::Id i, j, k, m, n;
  rbmd::Real sixth = 1.0 / 6.0;

  rbmd::Real rmax;
  eam_paras.dr = eam_paras.drho = rmax = eam_paras.rhomax = 0.0;

  eam_paras.dr = MAX(eam_paras.dr, file.dr);
  eam_paras.drho = MAX(eam_paras.drho, file.drho);
  rmax = MAX(rmax, (file.nr - 1) * file.dr);
  eam_paras.rhomax = MAX(eam_paras.rhomax, (file.nrho - 1) * file.drho);

  // set nr,nrho from cutoff and spacings
  // 0.5 is for round-off in divide

  eam_paras.nr = static_cast<int>(rmax / eam_paras.dr + 0.5);
  eam_paras.nrho = static_cast<int>(eam_paras.rhomax / eam_paras.drho + 0.5);

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------
  frho.resize(eam_paras.nrho + 1);

  rbmd::Real r, p, cof1, cof2, cof3, cof4;
  for (m = 1; m <= eam_paras.nrho; m++)
  {
    r = (m - 1) * eam_paras.drho;
    p = r / file.drho + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file.nrho - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    frho[m] = cof1 * file.frho[k - 1] + cof2 * file.frho[k] + cof3 * file.frho[k + 1] +
      cof4 * file.frho[k + 2];
  }

  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------
  rhor.resize(eam_paras.nrho + 1);
  for (m = 1; m <= eam_paras.nr; m++)
  {
    r = (m - 1) * eam_paras.dr;
    p = r / file.dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file.nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    auto cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    auto cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    auto cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    auto cof4 = sixth * p * (p * p - 1.0);
    rhor[m] = cof1 * file.rhor[k - 1] + cof2 * file.rhor[k] + cof3 * file.rhor[k + 1] +
      cof4 * file.rhor[k + 2];
  }

  // ------------------------------------------------------------------
  // setup z2r arrays
  // ------------------------------------------------------------------
  z2r.resize(eam_paras.nr + 1);

  double zri;
  for (m = 1; m <= eam_paras.nr; m++)
  {
    r = (m - 1) * eam_paras.dr;

    p = r / file.dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file.nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zri = cof1 * file.zr[k - 1] + cof2 * file.zr[k] + cof3 * file.zr[k + 1] + cof4 * file.zr[k + 2];

    z2r[m] = 27.2 * 0.529 * zri * zri;
  }
}

void EAM::array2spline() {
  _h_frho_spline.resize(eam_paras.nrho + 1);
  _h_rhor_spline.resize(eam_paras.nrho + 1);
  _h_z2r_spline.resize(eam_paras.nr + 1);

  interpolate(eam_paras.nrho, eam_paras.drho, frho, _h_frho_spline);
  interpolate(eam_paras.nr, eam_paras.dr, rhor, _h_rhor_spline);
  interpolate(eam_paras.nr, eam_paras.dr, z2r, _h_z2r_spline);
}

void EAM::interpolate(rbmd::Id n, rbmd::Real delta, std::vector<rbmd::Real>& f,
                       thrust::host_vector<Real7>& spline) {

  for (int m = 1; m <= n; m++)
  {
    spline[m][6] = f[m];  //f(x)
  }

  spline[1][5] = spline[2][6] -
    spline[1][6]; //f'(x) = (f(x + h) - f(x)) / h    [5] is the coefficient of the first derivative(energy expression)
  spline[2][5] = 0.5 * (spline[3][6] - spline[1][6]);
  spline[n - 1][5] = 0.5 * (spline[n][6] - spline[n - 2][6]);
  spline[n][5] = spline[n][6] - spline[n - 1][6];

  for (int m = 3; m <= n - 2; m++)
  {
    spline[m][5] =
      ((spline[m - 2][6] - spline[m + 2][6]) + 8.0 * (spline[m + 1][6] - spline[m - 1][6])) /
      12.0; //further sample points for a more accurate estimate
  }

  for (int m = 1; m <= n - 1; m++)
  {
    spline[m][4] = 3.0 * (spline[m + 1][6] - spline[m][6]) - 2.0 * spline[m][5] -
      spline[m + 1][5]; //[4] is the coefficient of the second derivative
    spline[m][3] = spline[m][5] + spline[m + 1][5] -
      2.0 * (spline[m + 1][6] - spline[m][6]); // [3] is the coefficient of the third derivative
  }

  spline[n][4] = 0.0;
  spline[n][3] = 0.0; //The second and third derivative coefficients at the last sample point are zero,
  // To make the interpolation curve smoother at both ends, the higher derivative coefficient at the boundary can be set to zero.
  // This is because spline interpolation typically uses higher-order polynomial interpolation
  // at the inner sample points and lower-order polynomials at the boundaries to ensure smoothness.

  for (int m = 1; m <= n; m++)
  {
    spline[m][2] = spline[m][5] / delta;       //The coefficient of the second derivative(force expression)
    spline[m][1] = 2.0 * spline[m][4] / delta; //The coefficient of the first derivative
    spline[m][0] = 3.0 * spline[m][3] / delta; //The coefficient of the zero derivative (i.e. the value of the function).
  }
}

void EAM::SetEAM() {

  _d_frho_spline.resize(eam_paras.nrho + 1);
  _d_rhor_spline.resize(eam_paras.nrho + 1);
  _d_z2r_spline.resize(eam_paras.nr + 1);

  //H2D
  thrust::copy(_h_frho_spline.begin(),
  _h_frho_spline.end(), _d_frho_spline.begin());

  thrust::copy(_h_rhor_spline.begin(),
_h_rhor_spline.end(), _d_rhor_spline.begin());

  thrust::copy(_h_z2r_spline.begin(),
_h_z2r_spline.end(), _d_z2r_spline.begin());
}

void EAM::EAMVerlet() {
    //neighbor_list_build
  auto start = std::chrono::high_resolution_clock::now();
  _list = _neighbor_list_builder->Build();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration = end - start;
  TimingStatistics::Instance().record("Neighbor-List",duration.count());

  //EAM_fp
  auto num_atoms = *(_structure_info_data->_num_atoms);
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  const auto total_atoms = linked_cell->_total_atoms_num;
  thrust::device_vector<rbmd::Real> eam_fp(total_atoms);

  thrust::device_vector<rbmd::Real> d_energy_embedding(1, 0.0);
  thrust::device_vector<rbmd::Real> d_energy_pair(1, 0.0);

  const char* validate_neighbors =
      std::getenv("RBMD_DEBUG_VALIDATE_EAM_VERLET_NEIGHBORS");
  if (validate_neighbors != nullptr && validate_neighbors[0] != '\0' &&
      validate_neighbors[0] != '0') {
    auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
    const rbmd::Id total_atoms = linked_cell ? linked_cell->_total_atoms_num : 0;
    thrust::host_vector<rbmd::Id> h_start = _list->_start_idx;
    thrust::host_vector<rbmd::Id> h_end = _list->_end_idx;
    thrust::host_vector<rbmd::Id> h_neighbors = _list->_d_neighbors;
    for (rbmd::Id atom = 0; atom < num_atoms; ++atom) {
      const auto atom_index = static_cast<std::size_t>(atom);
      const rbmd::Id begin = h_start[atom_index];
      const rbmd::Id end = h_end[atom_index];
      if (begin < 0 || end < begin ||
          static_cast<std::size_t>(end) > h_neighbors.size()) {
        std::ostringstream oss;
        oss << "Invalid EAM verlet neighbor range: atom=" << atom
            << " begin=" << begin << " end=" << end
            << " neighbors_size=" << h_neighbors.size()
            << " num_atoms=" << num_atoms
            << " total_atoms=" << total_atoms;
        throw std::runtime_error(oss.str());
      }
      for (rbmd::Id slot = begin; slot < end; ++slot) {
        const rbmd::Id neighbor = h_neighbors[static_cast<std::size_t>(slot)];
        if (neighbor < 0 || neighbor >= total_atoms) {
          std::ostringstream oss;
          oss << "Invalid EAM verlet neighbor index: atom=" << atom
              << " slot=" << slot << " neighbor=" << neighbor
              << " begin=" << begin << " end=" << end
              << " num_atoms=" << num_atoms
              << " total_atoms=" << total_atoms;
          throw std::runtime_error(oss.str());
        }
      }
    }
  }

  auto start_f = std::chrono::high_resolution_clock::now();
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::ComputeFpVerlet<device::DEVICE_GPU>()(
      force_box, eam_paras, file.cut_off, num_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(_list->_start_idx.data()),
      thrust::raw_pointer_cast(_list->_end_idx.data()),
      thrust::raw_pointer_cast(_list->_d_neighbors.data()),
      thrust::raw_pointer_cast(_d_rhor_spline.data()),
      thrust::raw_pointer_cast(_d_frho_spline.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(eam_fp.data()),
      thrust::raw_pointer_cast(d_energy_embedding.data()));
#ifdef USE_MPI
  GET_RBMD_PARALLEL->ForwardExchangeAtomScalar(eam_fp);
#endif
  op::ComputeEAMForceVerlet<device::DEVICE_GPU>()(
force_box, eam_paras ,file.cut_off, num_atoms,
thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
thrust::raw_pointer_cast(_list->_start_idx.data()),
thrust::raw_pointer_cast(_list->_end_idx.data()),
thrust::raw_pointer_cast(_list->_d_neighbors.data()),
thrust::raw_pointer_cast(_d_rhor_spline.data()),
thrust::raw_pointer_cast(_d_z2r_spline.data()),
thrust::raw_pointer_cast(_device_data->_d_px.data()),
thrust::raw_pointer_cast(_device_data->_d_py.data()),
thrust::raw_pointer_cast(_device_data->_d_pz.data()),
thrust::raw_pointer_cast(eam_fp.data()),
thrust::raw_pointer_cast(_device_data->_d_fx.data()),
thrust::raw_pointer_cast(_device_data->_d_fy.data()),
thrust::raw_pointer_cast(_device_data->_d_fz.data()),
thrust::raw_pointer_cast(d_energy_pair.data()));

  const char* dump_forces = std::getenv("RBMD_DEBUG_DUMP_EAM_FORCES");
  if (dump_forces != nullptr && dump_forces[0] != '\0' &&
      dump_forces[0] != '0') {
    thrust::host_vector<rbmd::Id> ids(num_atoms);
    thrust::host_vector<rbmd::Real> fx(num_atoms);
    thrust::host_vector<rbmd::Real> fy(num_atoms);
    thrust::host_vector<rbmd::Real> fz(num_atoms);
    thrust::copy_n(_device_data->_d_atoms_id.begin(), num_atoms, ids.begin());
    thrust::copy_n(_device_data->_d_fx.begin(), num_atoms, fx.begin());
    thrust::copy_n(_device_data->_d_fy.begin(), num_atoms, fy.begin());
    thrust::copy_n(_device_data->_d_fz.begin(), num_atoms, fz.begin());
    const auto filename = "eam_forces_rank" +
        std::to_string(rbmd::mpi::CurrentRank()) + ".csv";
    std::ofstream csv(filename, std::ios::app);
    if (csv.tellp() == 0) {
      csv << "step,gid,fx,fy,fz\n";
    }
    for (rbmd::Id i = 0; i < num_atoms; ++i) {
      csv << test_current_step << ',' << ids[i] << ',' << fx[i] << ','
          << fy[i] << ',' << fz[i] << '\n';
    }
  }

  // D2H
  thrust::host_vector<rbmd::Real> h_energy_embedding(d_energy_embedding);
  thrust::host_vector<rbmd::Real> h_energy_pair(d_energy_pair);
  _e_embedding = h_energy_embedding[0];
  _e_pair =  h_energy_pair[0];

  auto end_f = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration_f = end_f - start_f;
  TimingStatistics::Instance().record("Short-Range",duration_f.count());
}

void EAM::EAMDirectRBL() {
  const auto setup_start = std::chrono::high_resolution_clock::now();
  const auto linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
  PrepareDirectCellTraversal(linked_cell);
  const auto setup_end = std::chrono::high_resolution_clock::now();
  TimingStatistics::Instance().record(
      "Neighbor-List",
      std::chrono::duration<rbmd::Real>(setup_end - setup_start).count());

  const rbmd::Id num_atoms = *(_structure_info_data->_num_atoms);
  const rbmd::Id total_atoms = linked_cell->_total_atoms_num;
  if (num_atoms <= 0 || total_atoms < num_atoms) {
    throw std::runtime_error("Invalid atom counts for EAM direct traversal");
  }

  const op::EAMDirectTraversalData traversal{
      thrust::raw_pointer_cast(linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(linked_cell->_in_atom_list_end_index.data()),
      thrust::raw_pointer_cast(linked_cell->_cell_sorted_atom_indices.data()),
      linked_cell->GetDataPtr(), NeighborCellCount(*linked_cell),
      linked_cell->_cell_count_within_cutoff};

  const auto& config = DataManager::getInstance().getConfigData();
  const rbmd::Real r_core = config->Get<rbmd::Real>(
      "r_core", "hyper_parameters", "neighbor");
  const rbmd::Id neighbor_sample_num = config->Get<rbmd::Id>(
      "neighbor_sample_num", "hyper_parameters", "neighbor");
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  const rbmd::Id selection_frequency = ComputeRblSelectionFrequency(
      *linked_cell, force_box, r_core, file.cut_off, neighbor_sample_num);

  const auto force_start = std::chrono::high_resolution_clock::now();
  thrust::device_vector<rbmd::Real> eam_fp(total_atoms);
  thrust::device_vector<rbmd::Real> d_energy_embedding(1, 0.0);
  op::ComputeFpDirect<device::DEVICE_GPU>()(
      force_box, eam_paras, file.cut_off, num_atoms, traversal,
      thrust::raw_pointer_cast(_d_rhor_spline.data()),
      thrust::raw_pointer_cast(_d_frho_spline.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(eam_fp.data()),
      thrust::raw_pointer_cast(d_energy_embedding.data()));

#ifdef USE_MPI
  GET_RBMD_PARALLEL->ForwardExchangeAtomScalar(eam_fp);
#endif

  op::ComputeEAMForceRBLDirect<device::DEVICE_GPU>()(
      force_box, eam_paras, r_core, file.cut_off, num_atoms,
      neighbor_sample_num, selection_frequency, traversal,
      thrust::raw_pointer_cast(_d_rhor_spline.data()),
      thrust::raw_pointer_cast(_d_z2r_spline.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(eam_fp.data()),
      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
      thrust::raw_pointer_cast(_device_data->_d_fz.data()));

  const auto native_end_x = _device_data->_d_fx.begin() + num_atoms;
  const auto native_end_y = _device_data->_d_fy.begin() + num_atoms;
  const auto native_end_z = _device_data->_d_fz.begin() + num_atoms;
  _corr_value_x =
      thrust::reduce(_device_data->_d_fx.begin(), native_end_x, 0.0f,
                     thrust::plus<rbmd::Real>()) /
      num_atoms;
  _corr_value_y =
      thrust::reduce(_device_data->_d_fy.begin(), native_end_y, 0.0f,
                     thrust::plus<rbmd::Real>()) /
      num_atoms;
  _corr_value_z =
      thrust::reduce(_device_data->_d_fz.begin(), native_end_z, 0.0f,
                     thrust::plus<rbmd::Real>()) /
      num_atoms;
  op::FixRBLForceOp<device::DEVICE_GPU>()(
      num_atoms, _corr_value_x, _corr_value_y, _corr_value_z,
      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
      thrust::raw_pointer_cast(_device_data->_d_fz.data()));

  thrust::host_vector<rbmd::Real> h_energy_embedding(d_energy_embedding);
  _e_embedding = h_energy_embedding[0];
  _e_pair = 0;
  if (_energy_rbl_flag == "yes") {
    thrust::device_vector<rbmd::Real> d_energy_pair(1, 0.0);
    op::ComputeEAMEnergyDirect<device::DEVICE_GPU>()(
        force_box, eam_paras, file.cut_off, num_atoms, traversal,
        thrust::raw_pointer_cast(_d_rhor_spline.data()),
        thrust::raw_pointer_cast(_d_z2r_spline.data()),
        thrust::raw_pointer_cast(_device_data->_d_px.data()),
        thrust::raw_pointer_cast(_device_data->_d_py.data()),
        thrust::raw_pointer_cast(_device_data->_d_pz.data()),
        thrust::raw_pointer_cast(eam_fp.data()),
        thrust::raw_pointer_cast(d_energy_pair.data()));
    thrust::host_vector<rbmd::Real> h_energy_pair(d_energy_pair);
    _e_pair = h_energy_pair[0];
  }

  const auto force_end = std::chrono::high_resolution_clock::now();
  TimingStatistics::Instance().record(
      "Short-Range",
      std::chrono::duration<rbmd::Real>(force_end - force_start).count());
}

void EAM::EAMRBL() {

  // Build the full list first so rho/fp and RBL use one ghost snapshot.
  auto start_rbl = std::chrono::high_resolution_clock::now();
  _list = _neighbor_list_builder->Build();

  // compute force
  auto start_rbl_force = std::chrono::high_resolution_clock::now();

  auto num_atoms = *(_structure_info_data->_num_atoms);
  const auto total_atoms =
      LinkedCellLocator::GetInstance().GetLinkedCell()->_total_atoms_num;
  thrust::device_vector<rbmd::Real> eam_fp(total_atoms);
  thrust::device_vector<rbmd::Real> d_energy_embedding(1,0.0);

  //1:  compute fp
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::ComputeFpVerlet<device::DEVICE_GPU>()(
force_box, eam_paras ,file.cut_off, num_atoms,
thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
thrust::raw_pointer_cast(_list->_start_idx.data()),
thrust::raw_pointer_cast(_list->_end_idx.data()),
thrust::raw_pointer_cast(_list->_d_neighbors.data()),
thrust::raw_pointer_cast(_d_rhor_spline.data()),
thrust::raw_pointer_cast(_d_frho_spline.data()),
thrust::raw_pointer_cast(_device_data->_d_px.data()),
thrust::raw_pointer_cast(_device_data->_d_py.data()),
thrust::raw_pointer_cast(_device_data->_d_pz.data()),
thrust::raw_pointer_cast(eam_fp.data()),
thrust::raw_pointer_cast(d_energy_embedding.data()));

  thrust::host_vector<rbmd::Real> h_energy_embedding(d_energy_embedding);
  _e_embedding = h_energy_embedding[0];

#ifdef USE_MPI
  GET_RBMD_PARALLEL->ForwardExchangeAtomScalar(eam_fp);
#endif

  _rbl_list = _rbl_neighbor_list_builder->BuildFromCurrentGhosts();
  auto end_rbl = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration_rbl = end_rbl - start_rbl;
  TimingStatistics::Instance().record("Neighbor-List",duration_rbl.count());

 //2: compute EAM_RBL
  const auto r_core =
    DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
        "r_core", "hyper_parameters", "neighbor");
  const auto neighbor_sample_num =
  DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
      "neighbor_sample_num", "hyper_parameters", "neighbor");

  op::ComputeEAMForceRBL<device::DEVICE_GPU>()(
force_box, eam_paras ,r_core,file.cut_off, num_atoms,neighbor_sample_num,
_rbl_list->_selection_frequency,
thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
thrust::raw_pointer_cast(_rbl_list->_start_idx.data()),
thrust::raw_pointer_cast(_rbl_list->_end_idx.data()),
thrust::raw_pointer_cast(_rbl_list->_d_neighbors.data()),
thrust::raw_pointer_cast(_rbl_list->_d_random_neighbor.data()),
thrust::raw_pointer_cast(_rbl_list->_d_random_neighbor_num.data()),
thrust::raw_pointer_cast(_d_rhor_spline.data()),
thrust::raw_pointer_cast(_d_z2r_spline.data()),
thrust::raw_pointer_cast(_device_data->_d_px.data()),
thrust::raw_pointer_cast(_device_data->_d_py.data()),
thrust::raw_pointer_cast(_device_data->_d_pz.data()),
thrust::raw_pointer_cast(eam_fp.data()),
thrust::raw_pointer_cast(_device_data->_d_fx.data()),
thrust::raw_pointer_cast(_device_data->_d_fy.data()),
thrust::raw_pointer_cast(_device_data->_d_fz.data()));

  _corr_value_x =
    thrust::reduce(_device_data->_d_fx.begin(), _device_data->_d_fx.end(),
                   0.0f, thrust::plus<rbmd::Real>()) /num_atoms;
  _corr_value_y =
      thrust::reduce(_device_data->_d_fy.begin(), _device_data->_d_fy.end(),
                     0.0f, thrust::plus<rbmd::Real>()) /num_atoms;
  _corr_value_z =
      thrust::reduce(_device_data->_d_fz.begin(), _device_data->_d_fz.end(),
                     0.0f, thrust::plus<rbmd::Real>()) /num_atoms;
  // fix RBL:   rbl_force = force - corr_value
  op::FixRBLForceOp<device::DEVICE_GPU>()(
                      num_atoms, _corr_value_x, _corr_value_y, _corr_value_z,
                      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
                      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
                      thrust::raw_pointer_cast(_device_data->_d_fz.data()));

  auto end_rbl_force = std::chrono::high_resolution_clock::now();
  std::chrono::duration<rbmd::Real> duration_rbl_force = end_rbl_force - start_rbl_force;
  TimingStatistics::Instance().record("Short-Range",duration_rbl_force.count());


  //energy
  if ("yes" == _energy_rbl_flag ) {
    ComputEAMEnergy(eam_fp);
  }

}

void EAM::ComputEAMEnergy(
    const thrust::device_vector<rbmd::Real>& eam_fp) {
  // Reuse the full list that produced fp; rebuilding would replace ghosts.
  auto num_atoms = *(_structure_info_data->_num_atoms);
  thrust::device_vector<rbmd::Real> d_energy_pair(1, 0.0);

  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::ComputeEAMEnergy<device::DEVICE_GPU>()(
force_box, eam_paras ,file.cut_off, num_atoms,
thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
thrust::raw_pointer_cast(_list->_start_idx.data()),
thrust::raw_pointer_cast(_list->_end_idx.data()),
thrust::raw_pointer_cast(_list->_d_neighbors.data()),
thrust::raw_pointer_cast(_d_rhor_spline.data()),
thrust::raw_pointer_cast(_d_z2r_spline.data()),
thrust::raw_pointer_cast(_device_data->_d_px.data()),
thrust::raw_pointer_cast(_device_data->_d_py.data()),
thrust::raw_pointer_cast(_device_data->_d_pz.data()),
thrust::raw_pointer_cast(eam_fp.data()),
thrust::raw_pointer_cast(d_energy_pair.data()));

  // D2H
  thrust::host_vector<rbmd::Real> h_energy_pair(d_energy_pair);
  _e_pair = h_energy_pair[0];
}

void EAM::SumForces() {
  if (_direct_traversal) {
    EAMDirectRBL();
  } else if ("RBL" ==_neighbor_type) {
    EAMRBL();
  }
  else {
    EAMVerlet();
  }

  //
//   auto atom_id_to_idx =
// LinkedCellLocator::GetInstance().GetLinkedCell()->_atom_id_to_idx;
//    thrust::host_vector<rbmd::Real> h_fx(num_atoms);
//    thrust::host_vector<rbmd::Real> h_fy(num_atoms);
//    thrust::host_vector<rbmd::Real> h_fz(num_atoms);
//     h_fx = _device_data->_d_fx;
//     h_fy = _device_data->_d_fy;
//     h_fz = _device_data->_d_fz;
//
//    std::ofstream fx("fx.txt");
//    if (fx.is_open()) {
//      for (rbmd::Id i = 0; i < num_atoms; ++i) {
//        auto idx = atom_id_to_idx[i];
//        fx << i  << " " << h_fx[idx] << " " << h_fy[idx]
//          << " " << h_fz[idx]  << "\n";
//      }
//      fx.close();
//   }


}

void EAM::EvaluatePotentialEnergy() {

  _e_pe = GetGlobalRealSum(_e_embedding + _e_pair);
  //std::cout<<  "energy:  "<< _e_embedding   <<  ", "<<   _e_pair   <<   std::endl;
  ThermoStats::Instance().AddThermoData("total-potential-energy",_e_pe);

  //out
  auto interval = DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
"interval", "outputs", "thermo_out");
  if (!rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    return;
  }

  std::ofstream outfile("thermo.txt", std::ios::app);
  if (outfile.tellp() == 0) {
    outfile << "step  e_pe" << std::endl;
  }
  if (test_current_step % interval == 0) {
    outfile << test_current_step << " " <<  _e_pe << std::endl;
  }
  outfile.close();
}
