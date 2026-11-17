
#include "rbl_full_neighbor_list_builder.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
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

#include "common/revision_options.h"
#include "common/mpi_root_guard.hpp"
#include <thrust/host_vector.h>
#include <fstream>
#include <iomanip>

namespace {
void RecordBenchmarkRblSamples(const NeighborList& list, rbmd::Id n, rbmd::Id step) {
  const auto& options = rbmd::revision::GetOptions();
  // Accuracy-only reads must never enter the production timing path.
  if (!options.forces || !options.fixed_samples || step <= 0 || step % options.interval) return;
  if (list._rbl_fused_sampling || list._d_random_neighbor_num.size() < static_cast<std::size_t>(n))
    throw std::runtime_error("RBL sample diagnostics need a materialized owned-atom sample list");
  thrust::host_vector<rbmd::Id> counts(list._d_random_neighbor_num.begin(), list._d_random_neighbor_num.begin()+n);
  long double sum=0, square=0;
  rbmd::Id lo=n ? counts[0] : 0, hi=0;
  for (auto count : counts) {
    sum+=count; square+=static_cast<long double>(count)*count;
    lo=std::min(lo,count); hi=std::max(hi,count);
  }
  static std::ofstream out(options.prefix+".rbl_samples.rank"+std::to_string(rbmd::mpi::CurrentRank())+".csv");
  static bool header=false;
  if (!header) { out << "sample,n_owned,sum,sum_square,min,max\n"; header=true; }
  out << std::setprecision(20) << step << ',' << n << ',' << sum << ',' << square << ',' << lo << ',' << hi << '\n';
  out.flush();
  if (!out) throw std::runtime_error("Cannot write RBL sample diagnostics");
}
}

extern rbmd::Id test_current_step;

namespace {
using RblTimingClock = std::chrono::high_resolution_clock;

// Keep fixed-width candidate storage below the range where large Thrust/HIP
// vector indexing becomes fragile.  Dense systems retain the skin-based cell
// and halo reuse path, but sample RBL directly without materializing the full
// cutoff shell.
constexpr std::uint64_t kCandidateCacheEntryLimit =
    std::uint64_t{3} * (std::uint64_t{1} << 28);
constexpr std::uint64_t kImplicitCandidateCacheEntryLimit =
    std::uint64_t{3} * (std::uint64_t{1} << 30);
// The 96-entry cutoff was too permissive for dilute LJ: its conservative
// estimate selected an N x 128 dense u64 table even though the measured row
// maximum was only 48.  Reserve dense rows for genuinely dense systems and
// keep smaller rows on the compact CSR/append path.
constexpr rbmd::Id kDenseCandidateMinimumRowCapacity = rbmd::Id(256);

bool RblFusedFixedCoreEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("RBMD_RBL_FUSED_FIXED_CORE");
    // Keep the production path unchanged until membership/force A/B checks
    // have passed for both dense molecular benchmarks.
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
  }();
  return enabled;
}

bool RblFusedSamplingEnabled(rbmd::Id neighbor_sample_num,
                             rbmd::Id candidate_row_capacity) {
  (void)neighbor_sample_num;
  (void)candidate_row_capacity;
  // Pair's fused path samples the full skin candidate row and therefore does
  // not preserve the shell-only Bernoulli estimator used by cached/direct RBL.
  return false;
}

bool RblMemoryDiagnosticsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("RBMD_HIP_MEMORY_DIAG");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
  }();
  return enabled;
}

void LogRblHipMemory(const char* stage) {
  if (!RblMemoryDiagnosticsEnabled()) {
    return;
  }
  (void)DEVICESYNC();
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const ERROR_T status = MEMGETINFO(&free_bytes, &total_bytes);
  if (status != SUCCESS) {
    Logger::Instance().info("RBMD-HIP-MEM component=neighbor stage={} status={}",
                            stage, GETERRORSTRING(status));
    return;
  }
  Logger::Instance().info(
      "RBMD-HIP-MEM component=neighbor stage={} free_bytes={} used_bytes={} total_bytes={}",
      stage, free_bytes, total_bytes, total_bytes - free_bytes);
}

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

rbmd::Id RblSampleThreadsPerParticle(rbmd::Id max_draw_count) {
  static const rbmd::Id requested_threads_per_particle = [] {
    const char* value = std::getenv("RBMD_RBL_SAMPLE_TPP");
    if (value == nullptr || value[0] == '\0') {
      return rbmd::Id(0);
    }
    const int requested = std::atoi(value);
    switch (requested) {
      case 1:
      case 4:
      case 8:
      case 16:
      case 32:
        return static_cast<rbmd::Id>(requested);
      case 64:
        return static_cast<rbmd::Id>(WARP_SIZE);
      default:
        return rbmd::Id(0);
    }
  }();
  if (requested_threads_per_particle > 0) {
    return requested_threads_per_particle;
  }

  // Follow HOOMD's threads-per-particle strategy: match the logical subgroup
  // to the amount of candidate work available for one atom. This avoids idle
  // lanes for dilute water while exposing enough parallelism for polymer and
  // salt candidate rows.
  if (max_draw_count <= 4) {
    return rbmd::Id(4);
  }
  if (max_draw_count <= 8) {
    return rbmd::Id(8);
  }
  if (max_draw_count <= 16) {
    return rbmd::Id(16);
  }
  if (max_draw_count <= 32) {
    return rbmd::Id(32);
  }
  return rbmd::Id(WARP_SIZE);
}
}  // namespace

RblFullNeighborListBuilder::RblFullNeighborListBuilder() {
  LogRblHipMemory("constructor_entry");
  _r_core = DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
      "r_core", "hyper_parameters", "neighbor");
  // _linked_cell->_cell_count_within_cutoff = static_cast<rbmd::Id>(
  //     std::ceil(static_cast<double>(_linked_cell->_cutoff / _r_core)));
  // _linked_cell->Build(DataManager::getInstance().getMDData()->_h_box.get());
  // _linked_cell->SyncHToD();  // TODO: consider moving this synchronization into the locator.
  // _linked_cell->InitializeCells();
  _neighbor_sample_num =
      DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
          "neighbor_sample_num", "hyper_parameters", "neighbor");
  if (_neighbor_sample_num <= 0) {
    throw std::invalid_argument("Fixed-p RBL requires neighbor_sample_num > 0; p=0 has no unbiased nonempty-shell estimator");
  }
  _random_neighbor_capacity = _neighbor_sample_num;
  _d_required_random_neighbor_capacity.resize(1);
  _trunc_distance_power_2 = _r_core * _r_core - EPSILON;  // for estimate
  // Candidate-cache paths sample directly in Pair. Keep the O(N * sample)
  // materialized table empty unless the explicit fallback path needs it.
  _neighbor_list->_d_random_neighbor.clear();
  _neighbor_list->_d_random_neighbor_num.clear();
  LogRblHipMemory("constructor_no_random_table");
  const rbmd::Id active_atoms = ActiveAtomCount();
  if (!_candidate_list) {
    _candidate_list = std::make_shared<NeighborList>(active_atoms, false);
  } else {
    _candidate_list->resize(active_atoms);
  }
}

std::shared_ptr<NeighborList> RblFullNeighborListBuilder::Build() {
  return BuildImpl(true, {});
}

std::shared_ptr<NeighborList>
RblFullNeighborListBuilder::BuildWithStableAtoms(
    const std::function<void()>& on_stable_atoms,
    const std::function<void()>& on_native_atoms) {
  return BuildImpl(true, on_stable_atoms, on_native_atoms);
}

std::shared_ptr<NeighborList>
RblFullNeighborListBuilder::BuildFromCurrentGhosts() {
  return BuildImpl(false, {});
}

std::shared_ptr<NeighborList> RblFullNeighborListBuilder::BuildImpl(
    bool exchange_ghosts,
    const std::function<void()>& on_stable_atoms,
    const std::function<void()>& on_native_atoms) {
  _trunc_distance_power_2 = _r_core * _r_core - EPSILON;
  const rbmd::Real neighbor_cutoff = _linked_cell->_neighbor_cutoff;
  LogRblHipMemory("build_entry");
  const bool candidate_cache_enabled =
      _linked_cell->_skin > rbmd::Real(0) && SupportsCandidateCache();
  if (!candidate_cache_enabled) {
    throw std::runtime_error("FIXEDP_UNIFORM_PHYSICAL_SHELL_V3 requires the validated positive-skin full-candidate cache; direct Bernoulli fallback is forbidden");
  }
  const auto cache_check_start = RblTimingClock::now();
  // Direct traversal has no Verlet candidate shell to reuse. Reusing only the
  // displacement state here leaves cell and ghost indices stale.
  const bool reuse_neighbor_state =
      candidate_cache_enabled && ShouldReuseNeighborList(neighbor_cutoff);
  RecordRblTiming("RBL-CacheCheck", cache_check_start);
  if (reuse_neighbor_state) {
    LogRblHipMemory("reuse_entry");
    // Complete parameter collectives before the callback can broadcast RBE
    // samples. All ranks enter reuse through the same cache-validity reduction.
    GetRblParams();
    if (on_native_atoms) on_native_atoms();
#ifdef USE_MPI
    if (exchange_ghosts) {
      const auto exchange_start = RblTimingClock::now();
      GET_RBMD_PARALLEL->ForwardExchangeCoordinates();
      RecordRblTiming("RBL-CoordinateExchange", exchange_start);
    }
#endif
    const auto sample_start = RblTimingClock::now();
    if (on_stable_atoms) {
      on_stable_atoms();
    }
    if (candidate_cache_enabled) {
      _neighbor_list->_rbl_candidate_list = _candidate_list;
      const rbmd::Id candidate_row_capacity =
          _candidate_cache_max_per_atom > 0
              ? _candidate_cache_max_per_atom
              : _candidate_list->_max_neighbor_capacity;
      _neighbor_list->_rbl_fused_sampling = RblFusedSamplingEnabled(
          _neighbor_sample_num, candidate_row_capacity);
      _neighbor_list->_rbl_candidate_implicit_offsets =
          _candidate_cache_implicit_offsets;
      _neighbor_list->_rbl_sample_epoch = test_current_step;
      if (!_neighbor_list->_rbl_fused_sampling) {
        LogRblHipMemory("reuse_before_sample");
        SampleCandidateList();
        LogRblHipMemory("reuse_after_sample");
      }
    } else {
      _neighbor_list->_rbl_candidate_list.reset();
      _neighbor_list->_rbl_fused_sampling = false;
      while (GenerateNeighborsList() == RBMD_TRUE) {
        EstimateNeighborsList();
      }
    }
    RecordRblTiming("RBL-DirectSample", sample_start);
    RecordBenchmarkRblSamples(*_neighbor_list, ActiveAtomCount(), test_current_step);
  return _neighbor_list;
  }

  const auto rebuild_start = RblTimingClock::now();
  LogRblHipMemory("rebuild_entry");
#ifdef USE_MPI
  if (exchange_ghosts) {
    const auto rebuild_exchange_start = RblTimingClock::now();
    _linked_cell->ClearDataHalo();
    RbmdParallelUntilLocator::GetInstance()
        .GetRbmdParallelUntil()
        ->BalanceAndExchange();
    RecordRblTiming("RBL-RebuildExchange", rebuild_exchange_start);
  }
  if (this->_neighbor_list->_d_max_neighbor_num.size() <
      static_cast<std::size_t>(_linked_cell->_native_atoms_num)) {
    this->_neighbor_list->resize(_linked_cell->_native_atoms_num);
    _candidate_list->resize(_linked_cell->_native_atoms_num);
    this->should_realloc = true;
  }
#endif

  const auto assign_cells_start = RblTimingClock::now();
  _linked_cell->AssignAtomsToCell();
  RecordRblTiming("RBL-AssignCells", assign_cells_start);
  LogRblHipMemory("after_assign_cells");
  const auto sort_cells_start = RblTimingClock::now();
  _linked_cell->SortAtomsByCellKey();
  RecordRblTiming("RBL-SortCells", sort_cells_start);
  LogRblHipMemory("after_sort_cells");
  const auto cell_ranges_start = RblTimingClock::now();
  _linked_cell->ComputeCellRangesIndices();
  RecordRblTiming("RBL-CellRanges", cell_ranges_start);
  LogRblHipMemory("after_cell_ranges");
  const auto stencil_start = RblTimingClock::now();
  EnsureNeighborCellsCurrent();
  RecordRblTiming("RBL-Stencil", stencil_start);
  const auto params_start = RblTimingClock::now();
  GetRblParams();
  RecordRblTiming("RBL-Params", params_start);
  const rbmd::Id active_atoms = ActiveAtomCount();
  const rbmd::Id estimated_candidate_capacity =
      InitialNeighborCapacity(neighbor_cutoff);
  const std::uint64_t estimated_candidate_entries =
      static_cast<std::uint64_t>(active_atoms) *
      static_cast<std::uint64_t>(estimated_candidate_capacity);
  const bool candidate_cache_fits =
      estimated_candidate_entries <= kCandidateCacheEntryLimit;
  const bool dense_implicit_cache_fits =
      estimated_candidate_capacity >= kDenseCandidateMinimumRowCapacity &&
      estimated_candidate_entries <= kImplicitCandidateCacheEntryLimit;
  const bool compact_candidate_cache =
      candidate_cache_enabled && !candidate_cache_fits &&
      !dense_implicit_cache_fits;
  static bool candidate_cache_mode_reported = false;
  if (!candidate_cache_mode_reported) {
    Logger::Instance().info(
        "RBL candidate cache mode: {}, estimated entries: {}",
        compact_candidate_cache
            ? "compact"
            : (candidate_cache_enabled ? "cached" : "direct"),
        estimated_candidate_entries);
    candidate_cache_mode_reported = true;
  }
  if (on_stable_atoms) {
    on_stable_atoms();
  }
  _fixed_core_generated_this_build = false;
  if (candidate_cache_enabled) {
    if (should_realloc) {
      EstimateNeighborsList();
    }
    if (compact_candidate_cache) {
      if (!_compact_cache_seeded ||
          !TryGenerateAppendedCompactCandidateList()) {
        GenerateCompactCandidateList();
      }
      _compact_cache_seeded = true;
    } else if (dense_implicit_cache_fits ||
               _candidate_cache_max_per_atom >= rbmd::Id(96)) {
      const auto dense_fixed_start = RblTimingClock::now();
      // Dense rows are cheaper to rebuild in one fixed-width pass. The
      // low-density LJ path stays on guarded compact append below.
      if (!_candidate_cache_fixed_width) {
        const rbmd::Id capacity = std::max(
            estimated_candidate_capacity,
            GrownNeighborCapacity(_candidate_list->_max_neighbor_capacity));
        PrepareCandidateFixedStorage(active_atoms, capacity);
        _candidate_cache_fixed_width = true;
      }
      while (GenerateGuardedFixedCandidateAndCoreList() == RBMD_TRUE) {
        if (_required_candidate_capacity >
            _candidate_list->_max_neighbor_capacity) {
          PrepareCandidateFixedStorage(
              active_atoms,
              GrownNeighborCapacity(_required_candidate_capacity));
        }
        if (_required_neighbor_capacity >
            _neighbor_list->_max_neighbor_capacity) {
          EstimateNeighborsList();
        }
      }
      Logger::Instance().info(
          "RBL dense fixed-width rebuild: max per atom {}, capacity {}, "
          "implicit offsets {}",
          _required_candidate_capacity,
          _candidate_list->_max_neighbor_capacity,
          _candidate_cache_implicit_offsets ? 1 : 0);
      RecordRblTiming("RBL-DenseFixed", dense_fixed_start);
    } else {
      EstimateCandidateList();
      while (GenerateCandidateList() == RBMD_TRUE) {
        EstimateCandidateList();
      }
    }
    const auto deferred_filter_start = RblTimingClock::now();
    if (!_fixed_core_generated_this_build) {
      while (FilterCandidateList() == RBMD_TRUE) {
        EstimateNeighborsList();
      }
    }
    RecordRblTiming("RBL-DeferredFilter", deferred_filter_start);
    // Sort ONLY the full outer+skin candidate rows. Retain every alias; the
    // per-force sampler selects a current valid representative per global ID.
    // Original core+skin rows/force path stay unchanged and are validated here.
    {
      const Box fixedp_box = GetPeriodicBoxForNeighborAndForce(*_box);
      if (fixedp_box._type != Box::BoxType::ORTHOGONAL ||
          !fixedp_box._pbc_x || !fixedp_box._pbc_y || !fixedp_box._pbc_z) {
        throw std::runtime_error("Fixed-p v1 supports orthogonal fully periodic boxes only");
      }
      for (int axis=0;axis<3;++axis) {
        if (!(_linked_cell->_neighbor_cutoff < rbmd::Real(0.5)*fixedp_box._length[axis]))
          throw std::runtime_error("Fixed-p physical cutoff+skin must be below every periodic half-box length");
      }
      const auto force_field=DataManager::getInstance().getConfigData()->Get<std::string>(
          "type","hyper_parameters","force_field");
      if(force_field!="CVFF" && force_field!="LJ/CUT/COUL/LONG")
        throw std::runtime_error("Fixed-p v3 currently supports LJ/CUT/COUL/LONG and CVFF only");
      const bool cvff_distance=force_field=="CVFF";
      const rbmd::Id storage_atoms=_linked_cell->_total_atoms_num;
      if(storage_atoms<active_atoms) throw std::runtime_error("Fixed-p storage atoms < owned");
      const auto storage_size=static_cast<std::size_t>(storage_atoms);
      if(storage_size>_device_data->_d_atoms_id.size() || storage_size>_device_data->_d_atoms_type.size() ||
          storage_size>_device_data->_d_charge.size() || storage_size>_device_data->_d_px.size() ||
          storage_size>_device_data->_d_py.size() || storage_size>_device_data->_d_pz.size())
        throw std::runtime_error("Fixed-p owned/ghost storage exceeds attribute arrays");
      rbmd::Id error=0;
      CHECK_RUNTIME(MEMCPY(_d_should_realloc,&error,sizeof(rbmd::Id),H2D));
      op::PrepareRblFixedpRowsOp<device::DEVICE_GPU> prepare;
      const rbmd::Real core_superset=_r_core+_linked_cell->_skin;
      prepare(active_atoms,static_cast<rbmd::Id>(storage_size),
          thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
          thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
          thrust::raw_pointer_cast(_device_data->_d_charge.data()),
          thrust::raw_pointer_cast(_device_data->_d_px.data()),
          thrust::raw_pointer_cast(_device_data->_d_py.data()),
          thrust::raw_pointer_cast(_device_data->_d_pz.data()),
          thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
          thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
          thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
          thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
          _candidate_list->_max_neighbor_capacity,_candidate_cache_implicit_offsets,
          static_cast<std::uint64_t>(_candidate_list->_d_neighbors.size()),
          thrust::raw_pointer_cast(_neighbor_list->_start_idx.data()),
          thrust::raw_pointer_cast(_neighbor_list->_end_idx.data()),
          thrust::raw_pointer_cast(_neighbor_list->_d_neighbors.data()),
          static_cast<std::uint64_t>(_neighbor_list->_d_neighbors.size()),
          core_superset*core_superset+EPSILON,_r_core*_r_core,fixedp_box,cvff_distance,_d_should_realloc);
      CHECK_RUNTIME(MEMCPY(&error,_d_should_realloc,sizeof(rbmd::Id),D2H));

      if(error & rbmd::Id(128|2048)) {
        // Small failure-only witness; no trajectory or force changes, no guard bypass.
        thrust::device_vector<unsigned int> d_claim(1,0u);
        thrust::device_vector<rbmd::Id> d_wid(20,rbmd::Id(-1));
        thrust::device_vector<rbmd::Real> d_val(37,rbmd::Real(0));
        op::DiagnoseRblFixedpMissingCoreOp<device::DEVICE_GPU>()(
            active_atoms,storage_atoms,
            thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
            thrust::raw_pointer_cast(_device_data->_d_px.data()),
            thrust::raw_pointer_cast(_device_data->_d_py.data()),
            thrust::raw_pointer_cast(_device_data->_d_pz.data()),
            thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
            thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
            thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
            thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
            _candidate_list->_max_neighbor_capacity,_candidate_cache_implicit_offsets,
            static_cast<std::uint64_t>(_candidate_list->_d_neighbors.size()),
            thrust::raw_pointer_cast(_neighbor_list->_start_idx.data()),
            thrust::raw_pointer_cast(_neighbor_list->_end_idx.data()),
            thrust::raw_pointer_cast(_neighbor_list->_d_neighbors.data()),
            static_cast<std::uint64_t>(_neighbor_list->_d_neighbors.size()),
            core_superset*core_superset+EPSILON,_r_core*_r_core,fixedp_box,cvff_distance,
            thrust::raw_pointer_cast(d_claim.data()),thrust::raw_pointer_cast(d_wid.data()),
            thrust::raw_pointer_cast(d_val.data()));
        thrust::host_vector<unsigned int> h_claim=d_claim;
        thrust::host_vector<rbmd::Id> h_wid=d_wid;
        thrust::host_vector<rbmd::Real> h_val=d_val;
        const std::string path="fixedp_core_guard_witness.rank"+
            std::to_string(rbmd::mpi::CurrentRank())+".step"+std::to_string(test_current_step)+".json";
        std::ofstream witness(path);
        witness << std::setprecision(std::numeric_limits<rbmd::Real>::max_digits10)
          << "{\"witness_schema\":\"FIXEDP_CORE_ALIAS_INPUTS_V5\",\"status\":\"CACHE_OR_PHYSICAL_CORE_GUARD_REMAINS_FAILURE\",\"job\":\""
          << (std::getenv("SLURM_JOB_ID")?std::getenv("SLURM_JOB_ID"):"")
          << "\",\"rank\":" << rbmd::mpi::CurrentRank() << ",\"step\":" << test_current_step
          << ",\"error\":" << error << ",\"witness_found\":" << (h_claim[0]?"true":"false")
          << ",\"cvff_distance\":" << (cvff_distance?"true":"false") << ",\"ids\":[";
        for(std::size_t k=0;k<h_wid.size();++k){if(k)witness<<',';witness<<h_wid[k];}
        witness << "],\"values\":[";
        for(std::size_t k=0;k<h_val.size();++k){if(k)witness<<',';
          if(std::isfinite(h_val[k]))witness<<h_val[k];else witness<<"null";}
        witness << "]}\n";
        witness.close();
        Logger::Instance().info("Fixed-p core guard failure-only witness: {}",path);
      }

      if(error!=0) throw std::runtime_error("Fixed-p candidate/core validation error code "+std::to_string(error));
    }
    LogRblHipMemory("after_candidate_filter");
    _neighbor_list->_rbl_candidate_list = _candidate_list;
    // LJ, water, and molecular force styles consume candidate rows directly.
    const rbmd::Id candidate_row_capacity =
        _candidate_cache_max_per_atom > 0
            ? _candidate_cache_max_per_atom
            : _candidate_list->_max_neighbor_capacity;
    _neighbor_list->_rbl_fused_sampling = RblFusedSamplingEnabled(
        _neighbor_sample_num, candidate_row_capacity);
    _neighbor_list->_rbl_candidate_implicit_offsets =
        _candidate_cache_implicit_offsets;
    _neighbor_list->_rbl_sample_epoch = test_current_step;
    if (!_neighbor_list->_rbl_fused_sampling) {
      const auto rebuild_sample_start = RblTimingClock::now();
      LogRblHipMemory("rebuild_before_sample");
      SampleCandidateList();
      LogRblHipMemory("rebuild_after_sample");
      RecordRblTiming("RBL-RebuildSample", rebuild_sample_start);
    }
  } else {
    _neighbor_list->_rbl_candidate_list.reset();
    _neighbor_list->_rbl_fused_sampling = false;
    if (should_realloc) {
      EstimateNeighborsList();
    }
    while (GenerateNeighborsList() == RBMD_TRUE) {
      EstimateNeighborsList();
    }
  }
  RecordNeighborBuildState(neighbor_cutoff);
#ifdef USE_MPI
  // Preparing the forward ghost map is independent of whether the neighbour
  // list can be reused.  In direct (skin == 0) mode the halo is rebuilt every
  // step, but MPI SHAKE still needs a valid map for this step.  This also
  // remaps indices after BuildFromCurrentGhosts sorts the existing halo.
  const auto forward_setup_start = RblTimingClock::now();
  if (!GET_RBMD_PARALLEL->PrepareForwardCoordinateExchange()) {
    if (HasNeighborCache()) {
      InvalidateNeighborCache();
    }
  }
  RecordRblTiming("RBL-ForwardSetup", forward_setup_start);
#endif
  RecordRblTiming("RBL-CandidateRebuild", rebuild_start);
  RecordBenchmarkRblSamples(*_neighbor_list, ActiveAtomCount(), test_current_step);
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

  // Legacy scalar is not the fixed-p shell weight. The materialized per-owner
  // Real weight is current physical M/min(p,M), including fractional values.
  _selection_frequency = rbmd::Id(1);
  _neighbor_list->_selection_frequency = rbmd::Id(1);
#pragma endregion
}

void RblFullNeighborListBuilder::EstimateNeighborsList() {
  Logger::Instance().debug("Reallocating RBL core neighbor list capacity");
  const rbmd::Id capacity =
      _required_neighbor_capacity > _neighbor_list->_max_neighbor_capacity
          ? GrownNeighborCapacity(_required_neighbor_capacity)
          : std::max(_neighbor_list->_max_neighbor_capacity,
                     InitialNeighborCapacity(_r_core));
  PrepareFixedWidthList(_neighbor_list, ActiveAtomCount(), capacity);
  _required_neighbor_capacity = 0;
  this->should_realloc = false;
}

rbmd::Id RblFullNeighborListBuilder::GenerateNeighborsList() {
  GetRblParams();
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  CHECK_RUNTIME(
      MEMCPY(_d_should_realloc, &(this->should_realloc), sizeof(rbmd::Id), H2D));
  const rbmd::Id active_atoms = ActiveAtomCount();
  if (active_atoms == 0) {
    this->should_realloc = false;
    return RBMD_FALSE;
  }
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
    CHECK_RUNTIME(MEMCPY(
        thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()), &zero,
        sizeof(rbmd::Id), H2D));
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
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      _neighbor_cell_num,
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbor_num.data()),
      this->_neighbor_list->_max_neighbor_capacity,
      thrust::raw_pointer_cast(this->_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_end_idx.data()),
      thrust::raw_pointer_cast(this->_neighbor_list->_d_neighbors.data()),
      _random_neighbor_capacity,
      thrust::raw_pointer_cast(this->_neighbor_list->_d_random_neighbor.data()),
      thrust::raw_pointer_cast(
          this->_neighbor_list->_d_random_neighbor_num.data()),
      thrust::raw_pointer_cast(
          this->_neighbor_list->_d_random_neighbor_weight.data()),
      thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
      force_box, _d_should_realloc,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      _selection_frequency, test_current_step,
      ChooseThreadsPerParticle(_neighbor_list->_max_neighbor_capacity,
                               "RBMD_RBL_DIRECT_TPP", rbmd::Id(8)));
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
  CHECK_RUNTIME(MEMCPY(
      &_required_neighbor_capacity,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      sizeof(rbmd::Id), D2H));
  return this->should_realloc;
}

void RblFullNeighborListBuilder::EstimateCandidateList() {
  const rbmd::Id capacity =
      _required_candidate_capacity > _candidate_list->_max_neighbor_capacity
          ? GrownNeighborCapacity(_required_candidate_capacity)
          : std::max(_candidate_list->_max_neighbor_capacity,
                     InitialNeighborCapacity(_linked_cell->_neighbor_cutoff));
  PrepareFixedWidthList(_candidate_list, ActiveAtomCount(), capacity);
  _required_candidate_capacity = 0;
}

rbmd::Id RblFullNeighborListBuilder::GenerateCandidateList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  const rbmd::Id active_atoms = ActiveAtomCount();
  if (active_atoms == 0) {
    return RBMD_FALSE;
  }
  rbmd::Id candidate_realloc = RBMD_FALSE;
  CHECK_RUNTIME(MEMCPY(_d_should_realloc, &candidate_realloc, sizeof(rbmd::Id),
                       H2D));
  const rbmd::Id zero = 0;
  CHECK_RUNTIME(MEMCPY(
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()), &zero,
      sizeof(rbmd::Id), H2D));
  op::GenerateFullNeighborListOp<device::DEVICE_GPU> generate_candidates;
  // The 64-lane logical WarpScan path is unstable for dense candidate cells
  // on current DTK/HIP. Keep the cache generator on the validated 1/8-lane
  // paths; the legacy non-cache neighbor builder retains its own tuning.
  const rbmd::Id candidate_tpp = std::min(ChooseThreadsPerParticle(), rbmd::Id(8));
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
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      _neighbor_cell_num,
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      _candidate_list->_max_neighbor_capacity,
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
      force_box, _d_should_realloc,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      candidate_tpp);
  CHECK_RUNTIME(MEMCPY(&candidate_realloc, _d_should_realloc, sizeof(rbmd::Id),
                       D2H));
  CHECK_RUNTIME(MEMCPY(
      &_required_candidate_capacity,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      sizeof(rbmd::Id), D2H));
  return candidate_realloc;
}

void RblFullNeighborListBuilder::PrepareCandidateFixedStorage(
    rbmd::Id active_atoms, rbmd::Id capacity) {
  if (active_atoms < 0 || capacity <= 0) {
    throw std::runtime_error("Invalid implicit candidate-list dimensions");
  }
  const std::uint64_t storage_size =
      static_cast<std::uint64_t>(active_atoms) *
      static_cast<std::uint64_t>(capacity);
  if (storage_size >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("Implicit candidate storage exceeds size_t");
  }
  if (storage_size <= static_cast<std::uint64_t>(
                          std::numeric_limits<rbmd::Id>::max()) &&
      !_candidate_cache_implicit_offsets) {
    PrepareFixedWidthList(_candidate_list, active_atoms, capacity);
    return;
  }

  _candidate_cache_implicit_offsets = true;
  if (_candidate_list->_start_idx.size() <
      static_cast<std::size_t>(active_atoms)) {
    _candidate_list->resize(active_atoms);
  }
  _candidate_list->_max_neighbor_capacity = capacity;
  _candidate_list->_h_total_max_neighbor_num =
      std::numeric_limits<rbmd::Id>::max();
  _candidate_list->_d_neighbors.resize(
      static_cast<std::size_t>(storage_size));
  thrust::fill(_candidate_list->_d_neighbor_num.begin(),
               _candidate_list->_d_neighbor_num.begin() + active_atoms,
               rbmd::Id(0));
  Logger::Instance().info(
      "RBL implicit-64 candidate rows: entries {}, capacity {}",
      storage_size, capacity);
}

rbmd::Id RblFullNeighborListBuilder::GenerateGuardedFixedCandidateList() {
  return GenerateGuardedFixedCandidateAndCoreList();
}

rbmd::Id
RblFullNeighborListBuilder::GenerateGuardedFixedCandidateAndCoreList() {
  const rbmd::Id active_atoms = ActiveAtomCount();
  const bool fuse_core = RblFusedFixedCoreEnabled();
  if (active_atoms <= 0) {
    _required_candidate_capacity = 0;
    _required_neighbor_capacity = 0;
    _fixed_core_generated_this_build = fuse_core;
    return RBMD_FALSE;
  }
  const rbmd::Id candidate_capacity =
      _candidate_list->_max_neighbor_capacity;
  const std::uint64_t required_storage =
      static_cast<std::uint64_t>(active_atoms) *
      static_cast<std::uint64_t>(candidate_capacity);
  const bool storage_too_small =
      required_storage >
      static_cast<std::uint64_t>(_candidate_list->_d_neighbors.size());
  const bool offsets_no_longer_fit =
      !_candidate_cache_implicit_offsets &&
      required_storage > static_cast<std::uint64_t>(
                             std::numeric_limits<rbmd::Id>::max());
  const bool row_metadata_too_small =
      _candidate_list->_d_neighbor_num.size() <
      static_cast<std::size_t>(active_atoms);
  if (storage_too_small || offsets_no_longer_fit ||
      row_metadata_too_small) {
    PrepareCandidateFixedStorage(active_atoms, candidate_capacity);
  }
  const std::uint64_t required_core_storage =
      static_cast<std::uint64_t>(active_atoms) *
      static_cast<std::uint64_t>(_neighbor_list->_max_neighbor_capacity);
  if (_neighbor_list->_max_neighbor_capacity <= 0 ||
      _neighbor_list->_start_idx.size() <
          static_cast<std::size_t>(active_atoms) ||
      _neighbor_list->_end_idx.size() <
          static_cast<std::size_t>(active_atoms) ||
      _neighbor_list->_d_neighbor_num.size() <
          static_cast<std::size_t>(active_atoms) ||
      _neighbor_list->_d_neighbors.size() < required_core_storage) {
    EstimateNeighborsList();
  }
  const std::uint64_t allocated_core_storage =
      static_cast<std::uint64_t>(active_atoms) *
      static_cast<std::uint64_t>(_neighbor_list->_max_neighbor_capacity);
  if (fuse_core &&
      (_neighbor_list->_max_neighbor_capacity <= 0 ||
       _neighbor_list->_start_idx.size() <
           static_cast<std::size_t>(active_atoms) ||
       _neighbor_list->_end_idx.size() <
           static_cast<std::size_t>(active_atoms) ||
       _neighbor_list->_d_neighbor_num.size() <
           static_cast<std::size_t>(active_atoms) ||
       _neighbor_list->_d_neighbors.size() < allocated_core_storage)) {
    throw std::runtime_error("Unable to allocate fused RBL core storage");
  }
  _d_fixed_candidate_status.resize(5);
  thrust::fill(_d_fixed_candidate_status.begin(),
               _d_fixed_candidate_status.end(), rbmd::Id(0));
  op::GenerateRblFixedCandidatesOp<device::DEVICE_GPU> generate;
  const rbmd::Real core_superset_cutoff =
      _r_core + _linked_cell->_skin;
  const rbmd::Real core_cutoff_2 =
      fuse_core
          ? core_superset_cutoff * core_superset_cutoff + EPSILON
          : rbmd::Real(-1);
  generate(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff,
      core_cutoff_2,
      active_atoms, _linked_cell->_total_atoms_num, _linked_cell->_total_cells,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_px.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_py.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      _neighbor_cell_num, _candidate_list->_max_neighbor_capacity,
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      _candidate_cache_implicit_offsets,
      static_cast<std::uint64_t>(_candidate_list->_d_neighbors.size()),
      _neighbor_list->_max_neighbor_capacity,
      thrust::raw_pointer_cast(_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(_neighbor_list->_end_idx.data()),
      thrust::raw_pointer_cast(_neighbor_list->_d_neighbors.data()),
      thrust::raw_pointer_cast(_neighbor_list->_d_neighbor_num.data()),
      static_cast<std::uint64_t>(_neighbor_list->_d_neighbors.size()),
      GetPeriodicBoxForNeighborAndForce(*_box),
      thrust::raw_pointer_cast(_d_fixed_candidate_status.data()),
      ChooseThreadsPerParticle(candidate_capacity, "RBMD_RBL_FIXED_TPP",
                               rbmd::Id(8)));
  rbmd::Id status[5] = {0, 0, 0, 0, 0};
  CHECK_RUNTIME(MEMCPY(
      status, thrust::raw_pointer_cast(_d_fixed_candidate_status.data()),
      sizeof(status), D2H));
  _required_candidate_capacity = status[0];
  _required_neighbor_capacity = status[1];
  if (status[4] != 0) {
    throw std::runtime_error("Fused fixed candidate/core bounds error code " +
                             std::to_string(status[4]));
  }
  const rbmd::Id realloc = status[2] | status[3];
  _fixed_core_generated_this_build =
      fuse_core && status[2] == RBMD_FALSE && status[3] == RBMD_FALSE;
  Logger::Instance().info(
      "RBL fixed candidate rebuild: max per atom {}, capacity {}, "
      "core max {}, core capacity {}, fused core {}, implicit offsets {}",
      _required_candidate_capacity,
      _candidate_list->_max_neighbor_capacity,
      _required_neighbor_capacity,
      _neighbor_list->_max_neighbor_capacity,
      _fixed_core_generated_this_build ? 1 : 0,
      _candidate_cache_implicit_offsets ? 1 : 0);
  return realloc;
}

bool RblFullNeighborListBuilder::TryGenerateAppendedCompactCandidateList() {
  LogRblHipMemory("compact_append_entry");
  constexpr rbmd::Id append_max_per_atom = 96;
  const rbmd::Id active_atoms = ActiveAtomCount();
  if (_candidate_cache_max_per_atom >= append_max_per_atom) {
    // This method is entered only for the memory-driven compact mode.  Do not
    // silently convert a large compact cache back into the dense allocation
    // that compact mode was selected to avoid.  The caller will rebuild the
    // compact CSR with the validated count/scan/fill fallback instead.
    return false;
  }

  if (active_atoms <= 0) {
    _candidate_list->_d_neighbors.clear();
    return true;
  }
  const rbmd::Id previous_total =
      _candidate_list->_h_total_max_neighbor_num;
  if (previous_total <= 0 || _candidate_list->_d_neighbors.empty()) {
    return false;
  }
  const std::uint64_t previous =
      static_cast<std::uint64_t>(previous_total);
  const std::uint64_t headroom =
      std::max<std::uint64_t>(previous / 5, std::uint64_t{4096});
  const std::uint64_t id_limit = static_cast<std::uint64_t>(
      std::numeric_limits<rbmd::Id>::max());
  if (previous > id_limit - headroom) {
    return false;
  }
  const std::uint64_t storage_capacity = previous + headroom;

  _candidate_list->resize(active_atoms);
  _candidate_list->_d_neighbors.resize(
      static_cast<std::size_t>(storage_capacity));
  // Append overwrites count/start/end for every active row, including empty
  // rows. Avoid clearing three multi-million-entry metadata arrays first.

  const rbmd::Id zero = 0;
  CHECK_RUNTIME(MEMCPY(_d_should_realloc, &zero, sizeof(rbmd::Id), H2D));
  CHECK_RUNTIME(MEMCPY(
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()), &zero,
      sizeof(rbmd::Id), H2D));
  CHECK_RUNTIME(MEMCPY(
      thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
      &zero, sizeof(rbmd::Id), H2D));

  op::GenerateRblCompactCandidatesOp<device::DEVICE_GPU> append;
  append.Append(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff,
      active_atoms, _linked_cell->_total_atoms_num, _linked_cell->_total_cells,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_px.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_py.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      _neighbor_cell_num,
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
      storage_capacity,
      thrust::raw_pointer_cast(
          _d_required_random_neighbor_capacity.data()),
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      GetPeriodicBoxForNeighborAndForce(*_box), _d_should_realloc);

  rbmd::Id error_code = 0;
  rbmd::Id actual_total = 0;
  rbmd::Id max_per_atom = 0;
  CHECK_RUNTIME(MEMCPY(&error_code, _d_should_realloc, sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(MEMCPY(
      &actual_total,
      thrust::raw_pointer_cast(
          _d_required_random_neighbor_capacity.data()),
      sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(MEMCPY(
      &max_per_atom,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      sizeof(rbmd::Id), D2H));
  if (error_code != 0 || actual_total < 0 ||
      static_cast<std::uint64_t>(actual_total) > storage_capacity) {
    Logger::Instance().info(
        "RBL compact append fallback: error {}, entries {}, capacity {}",
        error_code, actual_total, storage_capacity);
    return false;
  }

  // Keep the validated headroom allocation. Consumers are bounded by the row
  // metadata and _h_total_max_neighbor_num, so shrinking here only forces the
  // next append to grow and value-initialize the same unused tail again.
  _candidate_list->_max_neighbor_capacity = max_per_atom;
  _candidate_cache_max_per_atom = max_per_atom;
  _candidate_list->_h_total_max_neighbor_num = actual_total;
  _candidate_cache_implicit_offsets = false;
  Logger::Instance().info(
      "RBL compact append entries: {}, max per atom: {}", actual_total,
      max_per_atom);
  LogRblHipMemory("compact_append_done");
  return true;
}

void RblFullNeighborListBuilder::GenerateCompactCandidateList() {
  LogRblHipMemory("compact_entry");
  const rbmd::Id active_atoms = ActiveAtomCount();
  if (active_atoms <= 0) {
    _candidate_list->_d_neighbors.clear();
    return;
  }
  _candidate_list->resize(active_atoms);
  thrust::fill(_candidate_list->_d_neighbor_num.begin(),
               _candidate_list->_d_neighbor_num.end(), rbmd::Id(0));
  thrust::fill(_candidate_list->_start_idx.begin(),
               _candidate_list->_start_idx.end(), rbmd::Id(0));
  thrust::fill(_candidate_list->_end_idx.begin(),
               _candidate_list->_end_idx.end(), rbmd::Id(0));
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  op::GenerateRblCompactCandidatesOp<device::DEVICE_GPU> generate_candidates;
  const rbmd::Id total_storage_atoms = _linked_cell->_total_atoms_num;
  const rbmd::Id total_cells = _linked_cell->_total_cells;
  if (total_storage_atoms < active_atoms || total_cells <= 0) {
    throw std::runtime_error("Invalid compact candidate storage dimensions");
  }

  rbmd::Id zero = 0;
  CHECK_RUNTIME(MEMCPY(_d_should_realloc, &zero, sizeof(rbmd::Id), H2D));
  CHECK_RUNTIME(MEMCPY(
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()), &zero,
      sizeof(rbmd::Id), H2D));
  generate_candidates.Count(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff,
      active_atoms, total_storage_atoms, total_cells,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_px.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_py.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      _neighbor_cell_num,
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()), force_box,
      _d_should_realloc);
  LogRblHipMemory("compact_after_count");

  rbmd::Id max_per_atom = 0;
  CHECK_RUNTIME(MEMCPY(
      &max_per_atom,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      sizeof(rbmd::Id), D2H));
  rbmd::Id error_code = 0;
  CHECK_RUNTIME(MEMCPY(&error_code, _d_should_realloc, sizeof(rbmd::Id), D2H));
  if (error_code != 0) {
    throw std::runtime_error("Compact candidate count bounds error code " +
                             std::to_string(error_code));
  }
  if (max_per_atom <= 0) {
    _candidate_list->_d_neighbors.clear();
    return;
  }
  _candidate_cache_max_per_atom = max_per_atom;
  const std::uint64_t total_candidates = thrust::reduce(
      _candidate_list->_d_neighbor_num.begin(),
      _candidate_list->_d_neighbor_num.end(), std::uint64_t{0});
  if (total_candidates <= 0) {
    throw std::overflow_error("Compact candidate count overflow");
  }
  if (total_candidates > static_cast<std::uint64_t>(
                             std::numeric_limits<rbmd::Id>::max())) {
    throw std::overflow_error("Compact candidate offsets exceed rbmd::Id");
  }
  thrust::exclusive_scan(_candidate_list->_d_neighbor_num.begin(),
                         _candidate_list->_d_neighbor_num.end(),
                         _candidate_list->_start_idx.begin());
  _candidate_list->_d_neighbors.resize(
      static_cast<std::size_t>(total_candidates));
  LogRblHipMemory("compact_after_storage_resize");
  _candidate_list->_max_neighbor_capacity = max_per_atom;
  _candidate_list->_h_total_max_neighbor_num =
      static_cast<rbmd::Id>(total_candidates);

  CHECK_RUNTIME(MEMCPY(_d_should_realloc, &zero, sizeof(rbmd::Id), H2D));
  generate_candidates.Fill(
      thrust::raw_pointer_cast(_linked_cell->_per_atom_cell_id.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_in_atom_list_start_index.data()),
      thrust::raw_pointer_cast(_linked_cell->_in_atom_list_end_index.data()),
      _linked_cell->_neighbor_cutoff * _linked_cell->_neighbor_cutoff,
      active_atoms, total_storage_atoms, total_cells,
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(
          _linked_cell->_cell_sorted_atom_indices.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_px.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_py.data()),
      thrust::raw_pointer_cast(_linked_cell->_cell_sorted_pz.data()),
      thrust::raw_pointer_cast(_linked_cell->_neighbor_cell.data()),
      _neighbor_cell_num,
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
      total_candidates, force_box, _d_should_realloc);
  LogRblHipMemory("compact_after_fill");
  CHECK_RUNTIME(MEMCPY(&error_code, _d_should_realloc, sizeof(rbmd::Id), D2H));
  if (error_code != 0) {
    throw std::runtime_error("Compact candidate fill bounds error code " +
                             std::to_string(error_code));
  }
  Logger::Instance().info(
      "RBL compact candidate cache entries: {}, max per atom: {}",
      total_candidates, max_per_atom);
}

rbmd::Id RblFullNeighborListBuilder::FilterCandidateList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  const rbmd::Id active_atoms = ActiveAtomCount();
  if (active_atoms == 0) {
    return RBMD_FALSE;
  }

  rbmd::Id core_realloc = RBMD_FALSE;
  CHECK_RUNTIME(
      MEMCPY(_d_should_realloc, &core_realloc, sizeof(rbmd::Id), H2D));
  op::FilterRblNeighborCandidatesOp<device::DEVICE_GPU> filter_candidates;
  const rbmd::Id zero = 0;
  CHECK_RUNTIME(MEMCPY(
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()), &zero,
      sizeof(rbmd::Id), H2D));
  const rbmd::Real core_superset_cutoff = _r_core + _linked_cell->_skin;
  filter_candidates(
      core_superset_cutoff * core_superset_cutoff + EPSILON, rbmd::Real(0),
      active_atoms,
      thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
      thrust::raw_pointer_cast(_device_data->_d_px.data()),
      thrust::raw_pointer_cast(_device_data->_d_py.data()),
      thrust::raw_pointer_cast(_device_data->_d_pz.data()),
      thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
      thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
      _candidate_list->_max_neighbor_capacity,
      _candidate_cache_implicit_offsets,
      thrust::raw_pointer_cast(_neighbor_list->_d_max_neighbor_num.data()),
      thrust::raw_pointer_cast(_neighbor_list->_start_idx.data()),
      thrust::raw_pointer_cast(_neighbor_list->_end_idx.data()),
      thrust::raw_pointer_cast(_neighbor_list->_d_neighbors.data()),
      thrust::raw_pointer_cast(_neighbor_list->_d_neighbor_num.data()),
      rbmd::Id(0), nullptr, nullptr, nullptr,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      force_box, _d_should_realloc, rbmd::Id(0), test_current_step,
      ChooseThreadsPerParticle(_candidate_list->_max_neighbor_capacity,
                               "RBMD_RBL_FILTER_TPP", rbmd::Id(8)));
  CHECK_RUNTIME(
      MEMCPY(&core_realloc, _d_should_realloc, sizeof(rbmd::Id), D2H));
  CHECK_RUNTIME(MEMCPY(
      &_required_neighbor_capacity,
      thrust::raw_pointer_cast(_d_required_neighbor_capacity.data()),
      sizeof(rbmd::Id), D2H));
  return core_realloc;
}

void RblFullNeighborListBuilder::SampleCandidateList() {
  const Box force_box = GetPeriodicBoxForNeighborAndForce(*_box);
  const rbmd::Id n=ActiveAtomCount();
  if(n==0) return;
  if(_neighbor_sample_num<=0) throw std::invalid_argument("Fixed-p requires positive p");
  if(!_candidate_list || _neighbor_list->_rbl_fused_sampling)
    throw std::runtime_error("Fixed-p requires materialized full candidate rows and fused sampling off");
  const auto force_field=DataManager::getInstance().getConfigData()->Get<std::string>(
      "type","hyper_parameters","force_field");
  if(force_field!="CVFF" && force_field!="LJ/CUT/COUL/LONG")
    throw std::runtime_error("Fixed-p v3 unsupported force field");
  const bool cvff_distance=force_field=="CVFF";
  const rbmd::Id storage_atoms=_linked_cell->_total_atoms_num;
  if(storage_atoms<n) throw std::runtime_error("Fixed-p storage atoms < owned");
  const auto storage_size=static_cast<std::size_t>(storage_atoms);
  if(storage_size>_device_data->_d_atoms_id.size() || storage_size>_device_data->_d_atoms_type.size() ||
      storage_size>_device_data->_d_charge.size() || storage_size>_device_data->_d_px.size() ||
      storage_size>_device_data->_d_py.size() || storage_size>_device_data->_d_pz.size())
    throw std::runtime_error("Fixed-p attribute bounds error");
  _neighbor_list->_d_random_neighbor_num.resize(n);
  _neighbor_list->_d_random_neighbor_population.resize(n);
  _neighbor_list->_d_random_neighbor_weight.resize(n);
  _random_neighbor_capacity=std::max(_random_neighbor_capacity,_neighbor_sample_num);
  op::SampleRblFixedpRowsOp<device::DEVICE_GPU> sample;
  while(true) {
    const std::uint64_t index_limit=std::min(std::uint64_t(std::numeric_limits<std::size_t>::max()),
                                           std::numeric_limits<std::uint64_t>::max());
    if(std::uint64_t(_random_neighbor_capacity)>index_limit/std::uint64_t(n))
      throw std::overflow_error("Fixed-p sample storage overflow before multiplication");
    const std::uint64_t storage=std::uint64_t(n)*std::uint64_t(_random_neighbor_capacity);
    if(_neighbor_list->_d_random_neighbor.size()<storage)
      _neighbor_list->_d_random_neighbor.resize(static_cast<std::size_t>(storage));
    rbmd::Id error=0,required=0;
    CHECK_RUNTIME(MEMCPY(_d_should_realloc,&error,sizeof(rbmd::Id),H2D));
    CHECK_RUNTIME(MEMCPY(thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
        &required,sizeof(rbmd::Id),H2D));
    sample(_r_core*_r_core,_linked_cell->_cutoff*_linked_cell->_cutoff,n,
        static_cast<rbmd::Id>(storage_size),
        thrust::raw_pointer_cast(_device_data->_d_atoms_id.data()),
        thrust::raw_pointer_cast(_device_data->_d_px.data()),
        thrust::raw_pointer_cast(_device_data->_d_py.data()),
        thrust::raw_pointer_cast(_device_data->_d_pz.data()),
        thrust::raw_pointer_cast(_candidate_list->_start_idx.data()),
        thrust::raw_pointer_cast(_candidate_list->_end_idx.data()),
        thrust::raw_pointer_cast(_candidate_list->_d_neighbors.data()),
        thrust::raw_pointer_cast(_candidate_list->_d_neighbor_num.data()),
        _candidate_list->_max_neighbor_capacity,_candidate_cache_implicit_offsets,
        _neighbor_sample_num,_random_neighbor_capacity,
        thrust::raw_pointer_cast(_neighbor_list->_d_random_neighbor.data()),
        thrust::raw_pointer_cast(_neighbor_list->_d_random_neighbor_num.data()),
        thrust::raw_pointer_cast(_neighbor_list->_d_random_neighbor_population.data()),
        thrust::raw_pointer_cast(_neighbor_list->_d_random_neighbor_weight.data()),
        thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
        force_box,cvff_distance,test_current_step,_d_should_realloc);
    CHECK_RUNTIME(MEMCPY(&error,_d_should_realloc,sizeof(rbmd::Id),D2H));
    CHECK_RUNTIME(MEMCPY(&required,thrust::raw_pointer_cast(_d_required_random_neighbor_capacity.data()),
        sizeof(rbmd::Id),D2H));
    if(error!=0) throw std::runtime_error("Fixed-p sampling error code "+std::to_string(error));
    if(required<=_random_neighbor_capacity) break;
    _random_neighbor_capacity=std::max(required,_random_neighbor_capacity*rbmd::Id(2));
  }
}
