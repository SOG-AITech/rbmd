
#include "rbl_full_neighbor_list_builder.h"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/scan.h>

#include "common/device_types.h"
#include "common/mpi_reduce_helper.hpp"
#include "force_box_selector.h"
#include "full_neighbor_list_op.h"
#include "neighbor_list_op.h"
#include "rbl_full_neighbor_list_op.h"
#include "common/timing_statistics.hpp"
#include "output/include/Logger.hpp"
#ifdef USE_MPI
#include "rbmd_parallel_until_locator.h"
#endif

extern rbmd::Id test_current_step;

namespace {
using RblTimingClock = std::chrono::high_resolution_clock;

void RecordRblTiming(const char* category,
                     RblTimingClock::time_point start) {
  static const bool enabled = [] {
    const char* value = std::getenv("RBMD_DETAILED_PAIR_TIMING");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
  }();
  if (!enabled) {
    return;
  }
  CHECK_RUNTIME(DEVICESYNC());
  const std::chrono::duration<double> duration =
      RblTimingClock::now() - start;
  TimingStatistics::Instance().record(category, duration.count());
}
}  // namespace

RblFullNeighborListBuilder::RblFullNeighborListBuilder() {
  _r_core = DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
      "r_core", "hyper_parameters", "neighbor");
  // _linked_cell->_cell_count_within_cutoff = static_cast<rbmd::Id>(
  //     std::ceil(static_cast<double>(_linked_cell->_cutoff / _r_core)));
  // _linked_cell->Build(DataManager::getInstance().getMDData()->_h_box.get());
  // _linked_cell->SyncHToD();  // TODO 逻辑有点复杂   是否放到locator里面去更好
  // _linked_cell->InitializeCells();
  _neighbor_sample_num =
      DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
          "neighbor_sample_num", "hyper_parameters", "neighbor");
  if (_neighbor_sample_num <= 0) {
    std::cout
        << "\033[31mError neighbor_sample_num must be large than 0.\033[0m"
        << std::endl;
    exit(0);
  }
  _random_neighbor_capacity = _neighbor_sample_num;
  _d_required_random_neighbor_capacity.resize(1);
  _trunc_distance_power_2 = _r_core * _r_core - EPSILON;  // for estimate
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  const unsigned long initial_random_neighbor_storage =
      static_cast<unsigned long>(active_atoms) *
      static_cast<unsigned long>(_neighbor_sample_num);
  _neighbor_list->_d_random_neighbor.resize(
      initial_random_neighbor_storage);  // cs neighbor
  _neighbor_list->_d_random_neighbor_num.resize(active_atoms);
  _candidate_list = std::make_shared<NeighborList>(active_atoms, false);
  // neighbor num !
  this->_neighbor_cell_num = (2 * _linked_cell->_cell_count_within_cutoff + 1) *
                             (2 * _linked_cell->_cell_count_within_cutoff + 1) *
                             (2 * _linked_cell->_cell_count_within_cutoff + 1);
}

std::shared_ptr<NeighborList> RblFullNeighborListBuilder::Build() {
  return BuildImpl(true);
}

std::shared_ptr<NeighborList>
RblFullNeighborListBuilder::BuildFromCurrentGhosts() {
  return BuildImpl(false);
}

std::shared_ptr<NeighborList> RblFullNeighborListBuilder::BuildImpl(
    bool exchange_ghosts) {
  _trunc_distance_power_2 = _r_core * _r_core - EPSILON;
  const rbmd::Real neighbor_cutoff = _linked_cell->_neighbor_cutoff;
  const bool candidate_cache_enabled =
      _linked_cell->_skin > rbmd::Real(0) && SupportsCandidateCache();
  const auto cache_check_start = RblTimingClock::now();
  const bool reuse_neighbor_state = ShouldReuseNeighborList(neighbor_cutoff);
  RecordRblTiming("RBL-CacheCheck", cache_check_start);
  if (reuse_neighbor_state) {
#ifdef USE_MPI
    if (exchange_ghosts) {
      const auto exchange_start = RblTimingClock::now();
      GET_RBMD_PARALLEL->ForwardExchangeCoordinates();
      RecordRblTiming("RBL-CoordinateExchange", exchange_start);
    }
#endif
    const auto filter_start = RblTimingClock::now();
    GetRblParams();
    if (candidate_cache_enabled && FilterCandidateList() == RBMD_TRUE) {
      EstimateNeighborsList();
      if (FilterCandidateList() == RBMD_TRUE) {
        throw std::runtime_error(
            "RBL core neighbor capacity remained insufficient after resize");
      }
    } else if (!candidate_cache_enabled &&
               GenerateNeighborsList() == RBMD_TRUE) {
      EstimateNeighborsList();
      GenerateNeighborsList();
    }
    RecordRblTiming("RBL-FilterSample", filter_start);
    return _neighbor_list;
  }

  const auto rebuild_start = RblTimingClock::now();
#ifdef USE_MPI
  if (exchange_ghosts) {
    _linked_cell->ClearDataHalo();
    RbmdParallelUntilLocator::GetInstance()
        .GetRbmdParallelUntil()
        ->BalanceAndExchange();
  }
  if (this->_neighbor_list->_d_max_neighbor_num.size() <
      static_cast<std::size_t>(_linked_cell->_native_atoms_num)) {
    this->_neighbor_list->resize(_linked_cell->_native_atoms_num);
    _candidate_list->resize(_linked_cell->_native_atoms_num);
    this->should_realloc = true;
  }
#endif

  this->_neighbor_cell_num = (2 * _linked_cell->_cell_count_within_cutoff + 1) *
                             (2 * _linked_cell->_cell_count_within_cutoff + 1) *
                             (2 * _linked_cell->_cell_count_within_cutoff + 1);
  _linked_cell->AssignAtomsToCell();
  _linked_cell->SortAtomsByCellKey();
  _linked_cell->ComputeCellRangesIndices();
  GetRblParams();
  if (candidate_cache_enabled) {
    if (should_realloc) {
      EstimateNeighborsList();
    }
    EstimateCandidateList();
    if (GenerateCandidateList() == RBMD_TRUE) {
      EstimateCandidateList();
      if (GenerateCandidateList() == RBMD_TRUE) {
        throw std::runtime_error(
            "RBL candidate capacity remained insufficient after resize");
      }
    }
    if (FilterCandidateList() == RBMD_TRUE) {
      EstimateNeighborsList();
      if (FilterCandidateList() == RBMD_TRUE) {
        throw std::runtime_error(
            "RBL core neighbor capacity remained insufficient after resize");
      }
    }
  } else {
    if (should_realloc) {
      EstimateNeighborsList();
    }
    if (GenerateNeighborsList() == RBMD_TRUE) {
      EstimateNeighborsList();
      GenerateNeighborsList();
    }
  }
  RecordNeighborBuildState(neighbor_cutoff);
#ifdef USE_MPI
  if (HasNeighborCache()) {
    if (!GET_RBMD_PARALLEL->PrepareForwardCoordinateExchange()) {
      InvalidateNeighborCache();
    }
  }
#endif
  RecordRblTiming("RBL-CandidateRebuild", rebuild_start);
  return _neighbor_list;
}

std::shared_ptr<NeighborList> RblFullNeighborListBuilder::Build(
    rbmd::Real custom_cutoff) {
  (void)custom_cutoff;
  return Build();
}

void RblFullNeighborListBuilder::GetRblParams() {
#pragma region rbl prarms
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  if (_global_atoms_num <= 0) {
    rbmd::Id local_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
    local_atoms = _linked_cell->_native_atoms_num;
#endif
    _global_atoms_num = GetGlobalIdSum(local_atoms);
  }
  const rbmd::Real volume = CalculateVolume(force_box);
  if (_global_atoms_num <= 0 || !(volume > rbmd::Real(0))) {
    throw std::runtime_error("Invalid RBL global density inputs");
  }
  _system_rho = static_cast<rbmd::Real>(_global_atoms_num) / volume;

  rbmd::Real coeff_rcs = 1.0 + (0.05 / _system_rho - 0.05);
  rbmd::Id Id_coeff_rcs = std::round(static_cast<double>(coeff_rcs));
  rbmd::Id rs_num = Id_coeff_rcs * _system_rho *
                        std::ceil(4.0 / 3.0 * M_PI * std::pow(_r_core, 3)) +
                    1;
  rbmd::Id rc_num =
      Id_coeff_rcs * _system_rho *
          std::ceil(4.0 / 3.0 * M_PI * std::pow(_linked_cell->_cutoff, 3)) +
      1;

  const rbmd::Id shell_num = rc_num - rs_num;
  _selection_frequency = shell_num <= _neighbor_sample_num
                             ? rbmd::Id(1)
                             : static_cast<rbmd::Id>(std::ceil(
                                   static_cast<rbmd::Real>(shell_num) /
                                   static_cast<rbmd::Real>(
                                       _neighbor_sample_num)));
  _neighbor_list->_selection_frequency = this->_selection_frequency;
#pragma endregion
}

void RblFullNeighborListBuilder::EstimateNeighborsList() {
  Logger::Instance().debug("Reallocating RBL core neighbor list capacity");
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  rbmd::Id* d_total_max_neighbor_num;
  CHECK_RUNTIME(MALLOC(&d_total_max_neighbor_num, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MEMCPY(d_total_max_neighbor_num,
                       &(_neighbor_list->_h_total_max_neighbor_num),
                       sizeof(rbmd::Id), H2D));
  op::EstimateRblFullNeighborListOp<device::DEVICE_GPU>
      estimate_rbl_full_neighbor_list_op;
  estimate_rbl_full_neighbor_list_op(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _trunc_distance_power_2, active_atoms,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(
          this->_neighbor_list->_d_max_neighbor_num.data()),
      force_box, _linked_cell->GetDataPtr(), _neighbor_cell_num,
      _linked_cell->_cell_count_within_cutoff);
  ReductionSum(
      thrust::raw_pointer_cast(_neighbor_list->_d_max_neighbor_num.data()),
      d_total_max_neighbor_num, active_atoms);
  CHECK_RUNTIME(MEMCPY(&(_neighbor_list->_h_total_max_neighbor_num),
                       d_total_max_neighbor_num, sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(FREE(d_total_max_neighbor_num));
  _neighbor_list->_d_neighbors.resize(
      _neighbor_list->_h_total_max_neighbor_num);
  InitNeighborListIndices();
  this->should_realloc = false;
}

rbmd::Id RblFullNeighborListBuilder::GenerateNeighborsList() {
  GetRblParams();
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  CHECK_RUNTIME(
      MEMCPY(_d_should_realloc, &(this->should_realloc), sizeof(rbmd::Id), H2D));
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  if (this->_neighbor_list->_d_random_neighbor_num.size() <
      static_cast<std::size_t>(active_atoms)) {
    this->_neighbor_list->_d_random_neighbor_num.resize(
        static_cast<std::size_t>(active_atoms));
  }
  op::GenerateRblFullNeighborListOp<device::DEVICE_GPU> generate_op;
  rbmd::Id required_capacity = 0;
  while (true) {
    const unsigned long required_storage =
        static_cast<unsigned long>(active_atoms) *
        static_cast<unsigned long>(_random_neighbor_capacity);
    if (_neighbor_list->_d_random_neighbor.size() < required_storage) {
      _neighbor_list->_d_random_neighbor.resize(required_storage);
    }
    const rbmd::Id zero = 0;
    CHECK_RUNTIME(MEMCPY(
        thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
        &zero, sizeof(rbmd::Id), H2D));
    generate_op(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _trunc_distance_power_2,
      _linked_cell->_cutoff * _linked_cell->_cutoff - EPSILON,
      active_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(
          this->_neighbor_list->_d_max_neighbor_num.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_end_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbors.data()),
      _random_neighbor_capacity,
      thrust::raw_pointer_cast(this->_neighbor_list->_d_random_neighbor.data()),
      thrust::raw_pointer_cast(
          this->_neighbor_list->_d_random_neighbor_num.data()),
      thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
      force_box, _d_should_realloc,
      _linked_cell->GetDataPtr(), _neighbor_cell_num, _selection_frequency,
      _linked_cell->_cell_count_within_cutoff, test_current_step);
    CHECK_RUNTIME(MEMCPY(
        &required_capacity,
        thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
        sizeof(rbmd::Id), D2H));
    if (required_capacity <= _random_neighbor_capacity) {
      break;
    }
    _random_neighbor_capacity =
        MAX(required_capacity, _random_neighbor_capacity * rbmd::Id(2));
  }
  CHECK_RUNTIME(
      MEMCPY(&(this->should_realloc), _d_should_realloc, sizeof(rbmd::Id), D2H));
  return this->should_realloc;
}

void RblFullNeighborListBuilder::EstimateCandidateList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  _candidate_list->resize(active_atoms);

  rbmd::Id* d_total_capacity = nullptr;
  CHECK_RUNTIME(MALLOC(&d_total_capacity, sizeof(rbmd::Id)));
  const rbmd::Id zero = 0;
  CHECK_RUNTIME(MEMCPY(d_total_capacity, &zero, sizeof(rbmd::Id), H2D));
  op::EstimateFullNeighborListOp<device::DEVICE_GPU> estimate_candidates;
  estimate_candidates(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff,
      active_atoms, thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_max_neighbor_num.data()),
      force_box, _linked_cell->GetDataPtr(), _neighbor_cell_num,
      _linked_cell->_cell_count_within_cutoff);
  ReductionSum(
      thrust::raw_pointer_cast(_candidate_list->_d_max_neighbor_num.data()),
      d_total_capacity, active_atoms);
  CHECK_RUNTIME(MEMCPY(&_candidate_list->_h_total_max_neighbor_num,
                       d_total_capacity, sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(FREE(d_total_capacity));

  _candidate_list->_d_neighbors.resize(
      _candidate_list->_h_total_max_neighbor_num);
  thrust::exclusive_scan(
      _candidate_list->_d_max_neighbor_num.begin(),
      _candidate_list->_d_max_neighbor_num.begin() + active_atoms,
      _candidate_list->_start_idx.begin());
  op::InitEndIndexOp<device::DEVICE_GPU> init_end;
  init_end(
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()), active_atoms);
}

rbmd::Id RblFullNeighborListBuilder::GenerateCandidateList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  rbmd::Id candidate_realloc = RBMD_FALSE;
  CHECK_RUNTIME(MEMCPY(_d_should_realloc, &candidate_realloc, sizeof(rbmd::Id),
                       H2D));
  op::GenerateFullNeighborListOp<device::DEVICE_GPU> generate_candidates;
  generate_candidates(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff,
      active_atoms, thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(
          _candidate_list->_d_max_neighbor_num.data()),
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()), force_box,
      _d_should_realloc, _linked_cell->GetDataPtr(), _neighbor_cell_num,
      _linked_cell->_cell_count_within_cutoff);
  CHECK_RUNTIME(MEMCPY(&candidate_realloc, _d_should_realloc, sizeof(rbmd::Id),
                       D2H));
  return candidate_realloc;
}

rbmd::Id RblFullNeighborListBuilder::FilterCandidateList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  rbmd::Id active_atoms = _linked_cell->_total_atoms_num;
#ifdef USE_MPI
  active_atoms = _linked_cell->_native_atoms_num;
#endif
  if (_neighbor_list->_d_random_neighbor_num.size() <
      static_cast<std::size_t>(active_atoms)) {
    _neighbor_list->_d_random_neighbor_num.resize(active_atoms);
  }

  rbmd::Id core_realloc = RBMD_FALSE;
  CHECK_RUNTIME(
      MEMCPY(_d_should_realloc, &core_realloc, sizeof(rbmd::Id), H2D));
  op::FilterRblNeighborCandidatesOp<device::DEVICE_GPU> filter_candidates;
  rbmd::Id required_capacity = 0;
  while (true) {
    const unsigned long required_storage =
        static_cast<unsigned long>(active_atoms) *
        static_cast<unsigned long>(_random_neighbor_capacity);
    if (_neighbor_list->_d_random_neighbor.size() < required_storage) {
      _neighbor_list->_d_random_neighbor.resize(required_storage);
    }
    const rbmd::Id zero = 0;
    CHECK_RUNTIME(MEMCPY(
        thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
        &zero, sizeof(rbmd::Id), H2D));
    filter_candidates(
      _r_core * _r_core - EPSILON,
      _linked_cell->_cutoff * _linked_cell->_cutoff - EPSILON, active_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
      thrust::raw_pointer_cast(_neighbor_list->_d_max_neighbor_num.data()),
      thrust::raw_pointer_cast(_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(_neighbor_list->_end_idx.data()),
      thrust::raw_pointer_cast(_neighbor_list->_d_neighbors.data()),
      _random_neighbor_capacity,
      thrust::raw_pointer_cast(_neighbor_list->_d_random_neighbor.data()),
      thrust::raw_pointer_cast(_neighbor_list->_d_random_neighbor_num.data()),
      thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
      force_box, _d_should_realloc, _selection_frequency, test_current_step);
    CHECK_RUNTIME(MEMCPY(
        &required_capacity,
        thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
        sizeof(rbmd::Id), D2H));
    if (required_capacity <= _random_neighbor_capacity) {
      break;
    }
    _random_neighbor_capacity =
        MAX(required_capacity, _random_neighbor_capacity * rbmd::Id(2));
  }
  CHECK_RUNTIME(
      MEMCPY(&core_realloc, _d_should_realloc, sizeof(rbmd::Id), D2H));
  return core_realloc;
}
