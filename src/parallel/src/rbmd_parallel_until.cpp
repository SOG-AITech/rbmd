#include "rbmd_parallel_until.h"

#include <sys/stat.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <unistd.h>

#include <climits>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "collective_communicator/collective_communicator.h"
#include "common/rbmd_define.h"
#include "common/startup_phase_debug.h"
#include "data_manager.h"
#include "mpi.h"
#include "mpi_data_type_utils.h"
#include "domain_decomposition/indirect_neighbour_communication_scheme.h"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"

extern rbmd::Id test_current_step;

namespace {

bool RuntimeOwnerAuditEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_RUNTIME_OWNER_AUDIT");
  if (env == nullptr) {
    return false;
  }
  return env[0] != '\0' && std::string(env) != "0";
}

bool NativeOwnershipInvariantAuditEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_NATIVE_OWNERSHIP_INVARIANTS");
  if (env == nullptr) {
    return false;
  }
  return env[0] != '\0' && std::string(env) != "0";
}

void EnsureRuntimeOwnerAuditDirectory() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  (void)mkdir("logs", 0755);
  (void)mkdir("logs/debug", 0755);
  initialized = true;
}

void AppendRuntimeOwnerAuditRow(const char* phase, rbmd::Id step,
                                const RbmdParallelUntil& parallel,
                                LinkedCell* linked_cell) {
  if (!RuntimeOwnerAuditEnabled() || phase == nullptr ||
      linked_cell == nullptr || !parallel._domdec) {
    return;
  }

  auto device_data = DataManager::getInstance().getDeviceData();
  if (!device_data) {
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

  thrust::copy(device_data->_d_atoms_id.begin(),
               device_data->_d_atoms_id.begin() + native_atoms, h_ids.begin());
  thrust::copy(device_data->_d_px.begin(),
               device_data->_d_px.begin() + native_atoms, h_px.begin());
  thrust::copy(device_data->_d_py.begin(),
               device_data->_d_py.begin() + native_atoms, h_py.begin());
  thrust::copy(device_data->_d_pz.begin(),
               device_data->_d_pz.begin() + native_atoms, h_pz.begin());

  const int current_rank = parallel._domdec->_current_rank;
  int mismatch_count = 0;
  rbmd::Id first_gid = -1;
  rbmd::Real first_px = 0;
  rbmd::Real first_py = 0;
  rbmd::Real first_pz = 0;
  int first_expected_owner = -1;
  rbmd::Id first_cell_idx = -1;
  int first_cell_is_halo = -1;

  for (std::size_t i = 0; i < native_n; ++i) {
    const int expected_owner =
        parallel.FindAtomOwnerRank(h_px[i], h_py[i], h_pz[i]);
    if (expected_owner == current_rank) {
      continue;
    }
    ++mismatch_count;
    if (first_gid >= 0) {
      continue;
    }
    first_gid = h_ids[i];
    first_px = h_px[i];
    first_py = h_py[i];
    first_pz = h_pz[i];
    first_expected_owner = expected_owner;
  }

  EnsureRuntimeOwnerAuditDirectory();
  std::ofstream csv("logs/debug/runtime_owner_audit_rank" +
                        std::to_string(current_rank) + "_pid" +
                        std::to_string(static_cast<int>(::getpid())) + ".csv",
                    std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,phase,rank,native_atoms,mismatch_count,first_gid,first_px,"
           "first_py,first_pz,first_expected_owner,first_cell_idx,"
           "first_cell_is_halo\n";
  }

  csv << step << "," << phase << "," << current_rank << "," << native_atoms
      << "," << mismatch_count << "," << first_gid << "," << first_px << ","
      << first_py << "," << first_pz << "," << first_expected_owner << ","
      << first_cell_idx << "," << first_cell_is_halo << "\n";
}

rbmd::Real OutsideLocalSubdomainDistance(const Box& local_box, rbmd::Real px,
                                         rbmd::Real py, rbmd::Real pz) {
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

void AppendNativeOwnershipInvariantRow(const char* phase, rbmd::Id step,
                                       const RbmdParallelUntil& parallel,
                                       LinkedCell* linked_cell) {
  if (!NativeOwnershipInvariantAuditEnabled() || phase == nullptr ||
      linked_cell == nullptr || !parallel._domdec) {
    return;
  }

  auto device_data = DataManager::getInstance().getDeviceData();
  auto local_box = DataManager::getInstance().getMDData()->_box.get();
  if (!device_data || local_box == nullptr) {
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
               device_data->_d_atoms_id.begin() + native_atoms, h_ids.begin());
  thrust::copy(device_data->_d_px.begin(),
               device_data->_d_px.begin() + native_atoms, h_px.begin());
  thrust::copy(device_data->_d_py.begin(),
               device_data->_d_py.begin() + native_atoms, h_py.begin());
  thrust::copy(device_data->_d_pz.begin(),
               device_data->_d_pz.begin() + native_atoms, h_pz.begin());

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
        local_box->_coord_min[0] <= px && px < local_box->_coord_max[0] &&
        local_box->_coord_min[1] <= py && py < local_box->_coord_max[1] &&
        local_box->_coord_min[2] <= pz && pz < local_box->_coord_max[2];
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
  }

  EnsureRuntimeOwnerAuditDirectory();
  const int current_rank = parallel._domdec->_current_rank;
  std::ofstream csv("logs/debug/native_ownership_invariants_rank" +
                        std::to_string(current_rank) + "_pid" +
                        std::to_string(static_cast<int>(::getpid())) + ".csv",
                    std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (csv.tellp() == 0) {
    csv << "step,phase,rank,native_atoms,native_outside_local_subdomain_count,"
           "native_in_halo_cell_count,shell_exceeded_count,max_outside_"
           "distance,"
           "halo_cutoff,first_outside_gid,first_halo_gid,"
           "first_shell_exceeded_gid\n";
  }

  csv << step << "," << phase << "," << current_rank << "," << native_atoms
      << "," << native_outside_local_subdomain_count << ","
      << native_in_halo_cell_count << "," << shell_exceeded_count << ","
      << max_outside_distance << "," << linked_cell->_halo_cutoff << ","
      << first_outside_gid << "," << first_halo_gid << ","
      << first_shell_exceeded_gid << "\n";
}

}  // namespace

void RbmdParallelUntil::Init(CollectiveCommunicator::Backend backend) {
  int current_rank = 0, total_ranks = 1;
  MPI_CHECK(MPI_Init(nullptr, nullptr));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &current_rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &total_ranks));
  this->_communicator = std::make_shared<CollectiveCommunicator>(
      backend, current_rank, total_ranks);
}

void RbmdParallelUntil::SetUp() {
  const auto& data_manager = DataManager::getInstance();
  const auto& config = data_manager.getConfigData();
  int current_rank = 0, total_ranks = 1;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &current_rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &total_ranks));
  rbmd::debug::StartupPhaseLog(
      "rbmd_parallel.setup.begin",
      "rank=" + std::to_string(current_rank) +
          " total_ranks=" + std::to_string(total_ranks));
  _domdec = std::make_shared<DomainDecomposition>(current_rank,
                                                  total_ranks);  // 处理盒子
  rbmd::debug::StartupPhaseLog(
      "rbmd_parallel.setup.after_domain_decomposition");

  _linked_cell = LinkedCellLocator::GetInstance()
                     .GetLinkedCell();  // 根据新盒子build  todo 用setlinkcell
  rbmd::debug::StartupPhaseLog("rbmd_parallel.setup.after_linked_cell");
  // 这样更好区分调用链
  if (config->PathExists({"hyper_parameters", "coulomb", "type"})) {
    const auto coulomb_type =
        config->Get<std::string>("type", "hyper_parameters", "coulomb");
    if (coulomb_type == "RBE" || coulomb_type == "RBSOG") {
      _rbe_p_number = config->Get<rbmd::Id>("coulomb_sample_num",
                                            "hyper_parameters", "coulomb");
      if (_rbe_p_number <= 0) {
        throw std::runtime_error(
            "The RBE/RBSOG sample number must be greater than 0, with a "
            "recommendation of being greater than 100.");
      }

      const bool has_configured_seed = config->PathExists(
          {"hyper_parameters", "coulomb", "random_seed"});
      unsigned int base_seed = 0u;
      if (has_configured_seed) {
        const auto configured_seed = config->Get<std::uint64_t>(
            "random_seed", "hyper_parameters", "coulomb");
        if (configured_seed >
            static_cast<std::uint64_t>(
                std::numeric_limits<unsigned int>::max())) {
          throw std::runtime_error(
              "hyper_parameters.coulomb.random_seed exceeds UINT_MAX");
        }
        base_seed = static_cast<unsigned int>(configured_seed);
      } else if (current_rank == 0) {
        base_seed = rd();
      }
      MPI_CHECK(MPI_Bcast(&base_seed, 1, MPI_UNSIGNED, 0,
                          _domdec->_mpi_comm));
      _rbe_base_seed = base_seed;
      _rbe_seed_generator.seed(_rbe_base_seed);
      _rbe_seed_sequence = 0;
      _rbe_seed_ready = true;
      if (current_rank == 0) {
        std::cout << "RBE/RBSOG random seed: " << _rbe_base_seed
                  << " (source="
                  << (has_configured_seed ? "config" : "random_device")
                  << ")" << std::endl;
      }
    }
  }
  MpiDataTypeUtils::GetInstance().CreateHaloAtomMpiType();
  rbmd::debug::StartupPhaseLog("rbmd_parallel.setup.after_halo_mpi_type");
  MpiDataTypeUtils::GetInstance().CreateLeavingAtomMpiType();
  rbmd::debug::StartupPhaseLog("rbmd_parallel.setup.after_leaving_mpi_type");
}

void RbmdParallelUntil::ForwardExchangeAtomScalar(
    thrust::device_vector<rbmd::Real>& values) {
  if (!_linked_cell) {
    throw std::runtime_error(
        "Cannot exchange atom scalar without linked cells");
  }

  const rbmd::Id native_atoms = _linked_cell->_native_atoms_num;
  const rbmd::Id total_atoms = _linked_cell->_total_atoms_num;
  if (native_atoms < 0 || total_atoms < native_atoms ||
      static_cast<std::size_t>(total_atoms) != values.size()) {
    throw std::runtime_error("Invalid atom counts for scalar forward exchange");
  }
  if (total_atoms == native_atoms) {
    return;
  }
  if (native_atoms > INT_MAX) {
    throw std::runtime_error("Too many native atoms for MPI scalar exchange");
  }

  auto device_data = DataManager::getInstance().getDeviceData();
  thrust::host_vector<rbmd::Id> local_ids(native_atoms);
  thrust::host_vector<rbmd::Real> local_values(native_atoms);
  thrust::copy_n(device_data->_d_atoms_id.begin(), native_atoms,
                 local_ids.begin());
  thrust::copy_n(values.begin(), native_atoms, local_values.begin());

  int rank_count = 0;
  MPI_CHECK(MPI_Comm_size(_domdec->_mpi_comm, &rank_count));
  const int local_count = static_cast<int>(native_atoms);
  std::vector<int> counts(rank_count, 0);
  MPI_CHECK(MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                          _domdec->_mpi_comm));

  std::vector<int> displacements(rank_count, 0);
  int global_count = 0;
  for (int rank = 0; rank < rank_count; ++rank) {
    displacements[rank] = global_count;
    if (counts[rank] < 0 || global_count > INT_MAX - counts[rank]) {
      throw std::runtime_error("Too many atoms for MPI scalar exchange");
    }
    global_count += counts[rank];
  }

  std::vector<rbmd::Id> global_ids(global_count);
  std::vector<rbmd::Real> global_values(global_count);
  MPI_CHECK(MPI_Allgatherv(
      local_ids.data(), local_count, MPI_RBMD_ID, global_ids.data(),
      counts.data(), displacements.data(), MPI_RBMD_ID, _domdec->_mpi_comm));
  MPI_CHECK(MPI_Allgatherv(
      local_values.data(), local_count, MPI_RBMD_REAL, global_values.data(),
      counts.data(), displacements.data(), MPI_RBMD_REAL, _domdec->_mpi_comm));

  std::unordered_map<rbmd::Id, rbmd::Real> owner_values;
  owner_values.reserve(global_ids.size());
  for (std::size_t i = 0; i < global_ids.size(); ++i) {
    if (!owner_values.emplace(global_ids[i], global_values[i]).second) {
      throw std::runtime_error(
          "Duplicate native atom owner in scalar exchange");
    }
  }

  thrust::host_vector<rbmd::Id> ghost_ids(total_atoms - native_atoms);
  thrust::copy(device_data->_d_atoms_id.begin() + native_atoms,
               device_data->_d_atoms_id.begin() + total_atoms,
               ghost_ids.begin());
  thrust::host_vector<rbmd::Real> ghost_values(ghost_ids.size());
  for (std::size_t i = 0; i < ghost_ids.size(); ++i) {
    const auto owner = owner_values.find(ghost_ids[i]);
    if (owner == owner_values.end()) {
      throw std::runtime_error("Missing ghost owner in scalar exchange");
    }
    ghost_values[i] = owner->second;
  }
  thrust::copy(ghost_values.begin(), ghost_values.end(),
               values.begin() + native_atoms);
}

bool RbmdParallelUntil::PrepareForwardCoordinateExchange() {
  if (!_linked_cell || !_domdec ||
      !_domdec->_neighbour_coneighbour_communication_scheme) {
    throw std::runtime_error(
        "Cannot prepare ghost coordinates without domain decomposition");
  }
  int comm_size = 1;
  MPI_CHECK(MPI_Comm_size(_domdec->_mpi_comm, &comm_size));
  if (comm_size <= 1) {
    return true;
  }
  const bool local_ready =
      _domdec->_neighbour_coneighbour_communication_scheme
          ->PrepareGhostCoordinateExchange(_linked_cell.get());
  int local = local_ready ? 1 : 0;
  int global = 0;
  MPI_CHECK(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                          _domdec->_mpi_comm));
  if (global == 0) {
    _domdec->_neighbour_coneighbour_communication_scheme
        ->InvalidateGhostCoordinateExchange();
    return false;
  }
  return true;
}

void RbmdParallelUntil::ForwardExchangeCoordinates() {
  if (!_linked_cell || !_domdec ||
      !_domdec->_neighbour_coneighbour_communication_scheme) {
    throw std::runtime_error(
        "Cannot exchange ghost coordinates without domain decomposition");
  }
  int comm_size = 1;
  MPI_CHECK(MPI_Comm_size(_domdec->_mpi_comm, &comm_size));
  if (comm_size <= 1) {
    return;
  }
  _domdec->_neighbour_coneighbour_communication_scheme
      ->ForwardGhostCoordinates(_linked_cell.get(), _domdec.get());
}

void RbmdParallelUntil::ForwardExchangeGhostState(ForwardGhostState state) {
  if (!_linked_cell || !_domdec ||
      !_domdec->_neighbour_coneighbour_communication_scheme) {
    throw std::runtime_error(
        "Cannot exchange ghost state without domain decomposition");
  }
  int comm_size = 1;
  MPI_CHECK(MPI_Comm_size(_domdec->_mpi_comm, &comm_size));
  if (comm_size <= 1) {
    return;
  }
  _domdec->_neighbour_coneighbour_communication_scheme->ForwardGhostFields(
      state, _linked_cell.get(), _domdec.get());
}

std::uint32_t RbmdParallelUntil::ForwardExchangeEpoch() const {
  if (!_domdec || !_domdec->_neighbour_coneighbour_communication_scheme) {
    return 0;
  }
  return _domdec->_neighbour_coneighbour_communication_scheme
      ->ForwardGhostPlanEpoch();
}

unsigned int RbmdParallelUntil::GetRbeGlobalRandomSeed() {
  if (!_rbe_seed_ready) {
    throw std::runtime_error(
        "RBE/RBSOG random seed generator is not initialized");
  }
  unsigned int random_seed = 0u;
  if (_domdec->_current_rank == 0) {
    random_seed = _rbe_seed_generator();
  }
  MPI_CHECK(MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0,
                      _domdec->_mpi_comm));
  ++_rbe_seed_sequence;
  return random_seed;
}

unsigned int RbmdParallelUntil::GetDeterministicRandomSeed() {
  return GetRbeGlobalRandomSeed();
}

int RbmdParallelUntil::FindAtomOwnerRank(rbmd::Real px, rbmd::Real py,
                                         rbmd::Real pz) const {
  static bool debug_printed = false;

  int owner_coords[3] = {0, 0, 0};
  // 计算原子在哪个网格区间
  // (coord - min) / (length / grid_dim)
  owner_coords[0] = static_cast<int>(
      (px - _global_structure_info.global_box._coord_min[0]) *
      _domdec->_grid_size[0] / _global_structure_info.global_box._length[0]);
  owner_coords[1] = static_cast<int>(
      (py - _global_structure_info.global_box._coord_min[1]) *
      _domdec->_grid_size[1] / _global_structure_info.global_box._length[1]);
  owner_coords[2] = static_cast<int>(
      (pz - _global_structure_info.global_box._coord_min[2]) *
      _domdec->_grid_size[2] / _global_structure_info.global_box._length[2]);

  // 【调试】打印第一个原子的计算过程
  if (!debug_printed) {
    std::cout << "===== FindAtomOwnerRank 调试（第一个原子）=====" << std::endl;
    std::cout << "原子位置: (" << px << ", " << py << ", " << pz << ")"
              << std::endl;
    std::cout << "盒子最小坐标: ("
              << _global_structure_info.global_box._coord_min[0] << ", "
              << _global_structure_info.global_box._coord_min[1] << ", "
              << _global_structure_info.global_box._coord_min[2] << ")"
              << std::endl;
    std::cout << "盒子长度: (" << _global_structure_info.global_box._length[0]
              << ", " << _global_structure_info.global_box._length[1] << ", "
              << _global_structure_info.global_box._length[2] << ")"
              << std::endl;
    std::cout << "网格大小: (" << _domdec->_grid_size[0] << ", "
              << _domdec->_grid_size[1] << ", " << _domdec->_grid_size[2] << ")"
              << std::endl;
    std::cout << "计算得到的网格坐标（边界处理前）: (" << owner_coords[0]
              << ", " << owner_coords[1] << ", " << owner_coords[2] << ")"
              << std::endl;
    debug_printed = true;
  }

  // 边界处理: 确保坐标不会越界
  for (int i = 0; i < 3; ++i) {
    if (owner_coords[i] < 0) owner_coords[i] = 0;
    if (owner_coords[i] >= _domdec->_grid_size[i])
      owner_coords[i] = _domdec->_grid_size[i] - 1;
  }

  int owner_rank;
  // 从逻辑网格坐标获取对应的进程 rank
  MPI_Cart_rank(_domdec->_mpi_comm, owner_coords, &owner_rank);

  return owner_rank;
}

void RbmdParallelUntil::GetRbeRadomM(
    rbmd::Real alpha, bool use_random, const Box& sample_box,
    const std::array<rbmd::Id, 3>& sample_multipliers,
    std::vector<rbmd::Real>& x, std::vector<rbmd::Real>& y,
    std::vector<rbmd::Real>& z) {
  x.resize(_rbe_p_number);
  y.resize(_rbe_p_number);
  z.resize(_rbe_p_number);
  const unsigned int sample_seed = GetRbeGlobalRandomSeed();
  std::mt19937 gen(sample_seed);

  // Local helper struct to mimic RBEPSample with deterministic generator
  struct RbeSampler {
    rbmd::Real _alpha;
    const Box* _box;
    int _P;
    bool _RBE_random;
    std::mt19937& _gen;

    RbeSampler(rbmd::Real alpha, const Box* box, int P, bool rbe_random,
               std::mt19937& gen)
        : _alpha(alpha), _box(box), _P(P), _RBE_random(rbe_random), _gen(gen) {}

    rbmd::Real RandomValue(const rbmd::Real& Min, const rbmd::Real& Max) {
      std::uniform_real_distribution<rbmd::Real> dis(Min, Max);
      return dis(_gen);
    }

    Real3 Compute_H() const {
      Real3 H;
      for (rbmd::Id i = 0; i < 3; ++i) {
        const rbmd::Real factor =
            -(_alpha * _box->_length[i] * _box->_length[i]);
        REAL_DATA(H)[i] = 0.0;

        for (rbmd::Id m = -10; m <= 10; m++) {
          rbmd::Real expx = m * m * factor;
          REAL_DATA(H)[i] += EXP(expx);
        }
        REAL_DATA(H)[i] *= SQRT(-(factor) / RBMD_PI);
      }

      return H;
    }

    rbmd::Real MH_Algorithm(rbmd::Real m, rbmd::Real mu, const Real3& sigma,
                            rbmd::Id dimension) {
      rbmd::Real x_wait = FetchSample_1D(mu, REAL_DATA(sigma)[dimension]);
      rbmd::Real m_wait = rbmd::Real(ROUND(x_wait));
      rbmd::Real Prob =
          (Distribution_P(m_wait, dimension) / Distribution_P(m, dimension)) *
          (Distribution_q(m, dimension) / Distribution_q(m_wait, dimension));
      Prob = MIN(Prob, rbmd::Real(1.0));

      if (_RBE_random) {
        rbmd::Real u = RandomValue(0.0, 1.0);
        if (u <= Prob) m = m_wait;
        return m;
      } else {
        rbmd::Real u = 0.5;
        if (u <= Prob) m = m_wait;
        return m;
      }
    }

    rbmd::Real Distribution_P(const rbmd::Real& x,
                              const rbmd::Id dimension) const {
      rbmd::Real P_m = EXP(-POW(2 * RBMD_PI * x / _box->_length[dimension], 2) /
                           (4 * _alpha));
      Real3 H = Compute_H();
      P_m = P_m / REAL_DATA(H)[dimension];
      return P_m;
    }

    rbmd::Real Distribution_q(const rbmd::Real& x,
                              const rbmd::Id dimension) const {
      rbmd::Real q_m;
      if (x == 0) {
        q_m = ERF((1.0 / 2) / (SQRT(_alpha * POW(_box->_length[dimension], 2) /
                                    POW(RBMD_PI, 2))));
      } else
        q_m = (ERF(((1.0 / 2) + ABS(x)) /
                   (SQRT(_alpha * POW(_box->_length[dimension], 2) /
                         POW(RBMD_PI, 2)))) -
               ERF((ABS(x) - (1.0 / 2)) /
                   (SQRT(_alpha * POW(_box->_length[dimension], 2) /
                         POW(RBMD_PI, 2))))) /
              2;
      return q_m;
    }

    rbmd::Real FetchSample_1D(
        const rbmd::Real& mu,
        const rbmd::Real&
            sigma) {  // Fetch 1D sample from Gaussion contribution
      rbmd::Real U1, U2, epsilon;
      epsilon = 1e-6;
      if (_RBE_random) {
        do {
          U1 = RandomValue(0.0, 1.0);
        } while (U1 < epsilon);
        U2 = RandomValue(0.0, 1.0);
        Real2 ChooseSample{0.0, 0.0};
        REAL_DATA(ChooseSample)
        [0] = sigma * SQRT(-2.0 * LOG(U1)) * COS(2 * RBMD_PI * U2) + mu;
        return REAL_DATA(ChooseSample)[0];
      } else {
        U1 = 0.5;
        U2 = 0.5;
        Real2 ChooseSample{0.0, 0.0};
        REAL_DATA(ChooseSample)
        [0] = sigma * SQRT(-2.0 * LOG(U1)) * COS(2 * RBMD_PI * U2) + mu;
        return REAL_DATA(ChooseSample)[0];
      }
    }

    void Fetch_P_Sample(const rbmd::Real& mu, const Real3& sigma,
                        std::vector<rbmd::Real>& P_Sample_x,
                        std::vector<rbmd::Real>& P_Sample_y,
                        std::vector<rbmd::Real>& P_Sample_z) {
      rbmd::Real epsilonx = 1e-6;  // precision
      Real3 X_0;
      do {
        X_0 = {rbmd::Real(ROUND(FetchSample_1D(mu, REAL_DATA(sigma)[0]))),
               rbmd::Real(ROUND(FetchSample_1D(mu, REAL_DATA(sigma)[1]))),
               rbmd::Real(ROUND(FetchSample_1D(mu, REAL_DATA(sigma)[2])))};
      } while (ABS(REAL_DATA(X_0)[0]) < epsilonx &&
               ABS(REAL_DATA(X_0)[1]) < epsilonx &&
               ABS(REAL_DATA(X_0)[2]) < epsilonx);
      /// 记录第一个样本
      P_Sample_x[0] = REAL_DATA(X_0)[0];
      P_Sample_y[0] = REAL_DATA(X_0)[1];
      P_Sample_z[0] = REAL_DATA(X_0)[2];

      //
      for (rbmd::Id i = 1; i < _P; i++) {
        Real3 X_1 = {MH_Algorithm(REAL_DATA(X_0)[0], mu, sigma, 0),
                     MH_Algorithm(REAL_DATA(X_0)[1], mu, sigma, 1),
                     MH_Algorithm(REAL_DATA(X_0)[2], mu, sigma, 2)};
        P_Sample_x[i] = REAL_DATA(X_1)[0];
        P_Sample_y[i] = REAL_DATA(X_1)[1];
        P_Sample_z[i] = REAL_DATA(X_1)[2];

        X_0 = X_1;
        if (ABS(REAL_DATA(X_1)[0]) < epsilonx &&
            ABS(REAL_DATA(X_1)[1]) < epsilonx &&
            ABS(REAL_DATA(X_1)[2]) < epsilonx) {
          i = i - 1;  //// 维持样本数量
          continue;
        }
      }
    }
  };

  RbeSampler sampler(alpha, &sample_box, _rbe_p_number, use_random, gen);

  Real3 sigma;
  REAL_DATA(sigma)
  [0] = SQRT(alpha * sample_box._length[0] * sample_box._length[0] /
             (2 * RBMD_PI * RBMD_PI));
  REAL_DATA(sigma)
  [1] = SQRT(alpha * sample_box._length[1] * sample_box._length[1] /
             (2 * RBMD_PI * RBMD_PI));
  REAL_DATA(sigma)
  [2] = SQRT(alpha * sample_box._length[2] * sample_box._length[2] /
             (2 * RBMD_PI * RBMD_PI));

  sampler.Fetch_P_Sample(0.0, sigma, x, y, z);
  if (sample_multipliers[0] > 1 || sample_multipliers[1] > 1 ||
      sample_multipliers[2] > 1) {
    for (int i = 0; i < _rbe_p_number; ++i) {
      x[i] *= sample_multipliers[0];
      y[i] *= sample_multipliers[1];
      z[i] *= sample_multipliers[2];
    }
  }
}

void RbmdParallelUntil::BalanceAndExchange() {
#ifdef USE_MPI
  if (_domdec && _domdec->_neighbour_coneighbour_communication_scheme) {
    _domdec->_neighbour_coneighbour_communication_scheme
        ->InvalidateGhostCoordinateExchange();
  }
  int comm_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if (comm_size <= 1) {
    return;
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  AppendRuntimeOwnerAuditRow("before_exchange", test_current_step, *this,
                             this->_linked_cell.get());
  _domdec->ExchangeMoleculesMPI(this->_linked_cell.get());
  AppendRuntimeOwnerAuditRow("after_exchange", test_current_step, *this,
                             this->_linked_cell.get());
  AppendNativeOwnershipInvariantRow("after_exchange", test_current_step, *this,
                                    this->_linked_cell.get());
}
