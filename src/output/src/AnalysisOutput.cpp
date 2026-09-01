#include "AnalysisOutput.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <experimental/filesystem>
#include <fstream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "Logger.hpp"
#include "common/mpi_root_guard.hpp"
#include "common/rbmd_define.h"
#include <thrust/copy.h>

namespace {
namespace fs = std::experimental::filesystem;

std::string SafeCurrentPath() {
  try {
    return fs::current_path().string();
  } catch (const std::exception&) {
    return "<unknown>";
  }
}

bool OpenOutputFile(std::ofstream& file, const std::string& path,
                    std::ios::openmode mode) {
  try {
    const auto status = fs::symlink_status(path);
    if (fs::exists(status) && !fs::is_regular_file(status)) {
      Logger::Instance().error(
          "Refuse to open analysis output file '{}' in cwd '{}': target is not a regular file.",
          path, SafeCurrentPath());
      return false;
    }
  } catch (const std::exception&) {
    Logger::Instance().error(
        "Failed to inspect analysis output target '{}' in cwd '{}'.", path,
        SafeCurrentPath());
    return false;
  }

  file.open(path, mode);
  if (!file.is_open()) {
    Logger::Instance().error(
        "Failed to open analysis output file '{}' in cwd '{}'.", path,
        SafeCurrentPath());
    return false;
  }
  return true;
}

template <typename T>
T* DataOrNull(std::vector<T>& values) {
  return values.empty() ? nullptr : values.data();
}

template <typename T>
const T* DataOrNull(const std::vector<T>& values) {
  return values.empty() ? nullptr : values.data();
}

Box BuildGlobalBoxFromLocalBounds(const std::vector<rbmd::Real>& gathered_bounds,
                                  const Box& local_box) {
  rbmd::Real coord_min[3] = {
      std::numeric_limits<rbmd::Real>::max(),
      std::numeric_limits<rbmd::Real>::max(),
      std::numeric_limits<rbmd::Real>::max()};
  rbmd::Real coord_max[3] = {
      std::numeric_limits<rbmd::Real>::lowest(),
      std::numeric_limits<rbmd::Real>::lowest(),
      std::numeric_limits<rbmd::Real>::lowest()};

  constexpr size_t kBoundsWidth = 6;
  const size_t num_ranks = gathered_bounds.size() / kBoundsWidth;
  for (size_t rank = 0; rank < num_ranks; ++rank) {
    const size_t base = rank * kBoundsWidth;
    for (int dim = 0; dim < 3; ++dim) {
      coord_min[dim] = std::min(coord_min[dim], gathered_bounds[base + dim]);
      coord_max[dim] =
          std::max(coord_max[dim], gathered_bounds[base + 3 + dim]);
    }
  }

  Box global_box = local_box;
  const bool pbc[3] = {local_box._pbc_x, local_box._pbc_y, local_box._pbc_z};
  global_box.Setup(local_box._type, coord_min, coord_max, pbc);
  return global_box;
}

rbmd::Real WrapCoordinateToPrimary(const Box& box, rbmd::Real coord, int dim,
                                   bool periodic) {
  if (!periodic || box._length[dim] <= rbmd::Real(0)) {
    return coord;
  }

  coord -= box._length[dim]
           * std::floor((coord - box._coord_min[dim]) * box._length_inv[dim]);

  if (coord >= box._coord_max[dim]) {
    coord -= box._length[dim];
  } else if (coord < box._coord_min[dim]) {
    coord += box._length[dim];
  }

  return coord;
}

std::string BuildRdfPath(const std::string& lhs_label,
                         const std::string& rhs_label) {
  return "rdf_" + lhs_label + "_" + rhs_label + ".txt";
}

rbmd::Id NormalizeOneBasedType(rbmd::Id type_value) {
  return type_value - 1;
}

size_t CountMatchingTypes(const std::vector<rbmd::Id>& types,
                          const std::vector<size_t>& type_counts) {
  size_t total = 0;
  for (const auto type_value : types) {
    if (type_value < 0) {
      continue;
    }
    const size_t index = static_cast<size_t>(type_value);
    if (index < type_counts.size()) {
      total += type_counts[index];
    }
  }
  return total;
}

void AddUniqueRdfPairIndex(std::vector<size_t>& pair_indices,
                           size_t pair_index) {
  if (std::find(pair_indices.begin(), pair_indices.end(), pair_index)
      == pair_indices.end()) {
    pair_indices.push_back(pair_index);
  }
}

size_t DetermineRdfWorkerThreads() {
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  const size_t detected = hardware_threads == 0
                              ? static_cast<size_t>(1)
                              : static_cast<size_t>(hardware_threads);
  return std::max(static_cast<size_t>(1),
                  std::min(detected, static_cast<size_t>(8)));
}

constexpr double kFourPiOverThree = 4.1887902047863905;

void ResolveRdfPairsFromLegacyAtomsPair(
    const RdfConfig& rdf_config, size_t rdf_num_bins,
    std::vector<AnalysisOutput::RdfPairState>& rdf_pairs) {
  rdf_pairs.clear();
  rdf_pairs.reserve(rdf_config.atoms_pair.size());
  for (const auto& atoms_pair : rdf_config.atoms_pair) {
    if (atoms_pair.size() < 2) {
      continue;
    }
    const rbmd::Id lhs_config_type = atoms_pair[0];
    const rbmd::Id rhs_config_type = atoms_pair[1];
    if (lhs_config_type < 1 || rhs_config_type < 1) {
      continue;
    }

    AnalysisOutput::RdfPairState pair_state;
    pair_state.lhs_label = std::to_string(lhs_config_type);
    pair_state.rhs_label = std::to_string(rhs_config_type);
    pair_state.lhs_types = {NormalizeOneBasedType(lhs_config_type)};
    pair_state.rhs_types = {NormalizeOneBasedType(rhs_config_type)};
    pair_state.self_pair =
        pair_state.lhs_types.front() == pair_state.rhs_types.front();
    pair_state.path = BuildRdfPath(pair_state.lhs_label, pair_state.rhs_label);
    pair_state.counts.assign(rdf_num_bins, 0.0);
    rdf_pairs.push_back(std::move(pair_state));
  }
}

void ResolveRdfPairsFromGroups(
    const RdfConfig& rdf_config, size_t rdf_num_bins,
    std::vector<AnalysisOutput::RdfPairState>& rdf_pairs) {
  rdf_pairs.clear();

  std::unordered_map<std::string, std::vector<rbmd::Id>> group_types;
  group_types.reserve(rdf_config.groups.size());
  for (const auto& group : rdf_config.groups) {
    if (group.name.empty() || group.types.empty()) {
      continue;
    }
    group_types[group.name] = group.types;
  }

  rdf_pairs.reserve(rdf_config.group_pairs.size());
  for (const auto& group_pair : rdf_config.group_pairs) {
    const auto lhs_it = group_types.find(group_pair.lhs);
    const auto rhs_it = group_types.find(group_pair.rhs);
    if (lhs_it == group_types.end() || rhs_it == group_types.end()) {
      continue;
    }

    AnalysisOutput::RdfPairState pair_state;
    pair_state.lhs_label = group_pair.lhs;
    pair_state.rhs_label = group_pair.rhs;
    pair_state.lhs_types = lhs_it->second;
    pair_state.rhs_types = rhs_it->second;
    pair_state.self_pair = group_pair.lhs == group_pair.rhs;
    pair_state.path = BuildRdfPath(pair_state.lhs_label, pair_state.rhs_label);
    pair_state.counts.assign(rdf_num_bins, 0.0);
    rdf_pairs.push_back(std::move(pair_state));
  }
}
}  // namespace

bool AnalysisOutput::ShouldSample(rbmd::Id interval, rbmd::Id step) const {
  return interval > 0 && step % interval == 0;
}

AnalysisOutput::AnalysisSchedule AnalysisOutput::BuildSchedule(
    rbmd::Id step) const {
  AnalysisSchedule schedule;
  schedule.rdf =
      _rdf_config.enabled && ShouldSample(_rdf_config.interval, step);
  schedule.msd = _msd_config.enabled &&
                 ShouldSample(_msd_config.interval, step) &&
                 step >= _msd_config.start_step && step <= _msd_config.end_step;
  schedule.vacf =
      _vacf_config.enabled && ShouldSample(_vacf_config.interval, step) &&
      step >= _vacf_config.start_step && step <= _vacf_config.end_step;
  return schedule;
}

AnalysisOutput::~AnalysisOutput() {
  _stop_flag.store(true);
  _cv_not_empty.notify_all();
  _cv_not_full.notify_all();

  if (_output_thread.joinable()) {
    _output_thread.join();
  }

  if (_stream) {
    CHECK_RUNTIME(STREAM_SYNC(_stream));
  }

  DeallocateRingBuffer();

  if (_stream) {
    CHECK_RUNTIME(STREAM_DESTORY(_stream));
    _stream = nullptr;
  }
}

void AnalysisOutput::Init() {
  auto config = DataManager::getInstance().getConfigData();
  _mpi_root_only_mode = false;
  _mpi_world_size = 1;
  _rdf_config = {};
  _msd_config = {};
  _vacf_config = {};
  _should_write_files = true;
  _simulation_timestep = rbmd::Real(1);
  _rdf_flush_interval = 0;
  _rdf_num_bins = 0;
  _rdf_worker_threads = DetermineRdfWorkerThreads();
  _rdf_bin_centers.clear();
  _rdf_shell_volumes.clear();
  _rdf_pairs.clear();
  _pending_frames.clear();
  _msd_state = {};
  _vacf_state = {};
  _num_atoms = CurrentLocalAtomCount();
  _write_index.store(0);
  _read_index.store(0);
  _stop_flag.store(false);

  if (config->PathExists({"execution", "timestep"})) {
    _simulation_timestep =
        config->Get<rbmd::Real>("timestep", "execution");
  }

#ifdef USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
      int mpi_world_size = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
      _mpi_world_size = mpi_world_size;
      _mpi_root_only_mode = mpi_world_size > 1;
      if (_mpi_root_only_mode) {
        _should_write_files = rbmd::mpi::ShouldWriteRootOnlyOutput();
      }
    }
  }
#endif

  if (!config->PathExists({"outputs"})) {
    _initialized = true;
    return;
  }

  auto& outputs = config->GetJsonNode("outputs");
  if (outputs.contains("rdf_out") && outputs["rdf_out"].is_object()) {
    auto& rdf_out = outputs["rdf_out"];
    _rdf_config.enabled = true;
    _rdf_config.interval = rdf_out.value("interval", rbmd::Id{0});
    _rdf_config.radius = rdf_out.value("radius", rbmd::Real{0});
    _rdf_config.dr = rdf_out.value("dr", rbmd::Real{0});
    _rdf_config.statistics_rdf_steps =
        rdf_out.value("statistics_rdf_steps", rbmd::Id{0});

    if (rdf_out.contains("atoms_pair") && rdf_out["atoms_pair"].is_array()) {
      _rdf_config.atoms_pair.clear();
      for (const auto& atoms_pair : rdf_out["atoms_pair"]) {
        if (!atoms_pair.is_array()) {
          continue;
        }
        _rdf_config.atoms_pair.push_back(
            atoms_pair.get<std::vector<rbmd::Id>>());
      }
    }

    if (rdf_out.contains("groups") && rdf_out["groups"].is_array()) {
      _rdf_config.groups.clear();
      for (const auto& group_json : rdf_out["groups"]) {
        if (!group_json.is_object() || !group_json.contains("name")
            || !group_json["name"].is_string()
            || !group_json.contains("types")
            || !group_json["types"].is_array()) {
          continue;
        }

        RdfGroupConfig group_config;
        group_config.name = group_json["name"].get<std::string>();
        for (const auto& type_json : group_json["types"]) {
          if (!type_json.is_number_integer()) {
            continue;
          }
          const rbmd::Id type_value = type_json.get<rbmd::Id>();
          if (type_value < 1) {
            continue;
          }
          group_config.types.push_back(NormalizeOneBasedType(type_value));
        }

        std::sort(group_config.types.begin(), group_config.types.end());
        group_config.types.erase(
            std::unique(group_config.types.begin(), group_config.types.end()),
            group_config.types.end());
        if (!group_config.name.empty() && !group_config.types.empty()) {
          _rdf_config.groups.push_back(std::move(group_config));
        }
      }
    }

    if (rdf_out.contains("group_pairs") && rdf_out["group_pairs"].is_array()) {
      _rdf_config.group_pairs.clear();
      for (const auto& pair_json : rdf_out["group_pairs"]) {
        if (!pair_json.is_array() || pair_json.size() < 2
            || !pair_json[0].is_string() || !pair_json[1].is_string()) {
          continue;
        }

        RdfGroupPairConfig group_pair;
        group_pair.lhs = pair_json[0].get<std::string>();
        group_pair.rhs = pair_json[1].get<std::string>();
        if (!group_pair.lhs.empty() && !group_pair.rhs.empty()) {
          _rdf_config.group_pairs.push_back(std::move(group_pair));
        }
      }
    }
  }

  if (outputs.contains("msd_out") && outputs["msd_out"].is_object()) {
    auto& msd_out = outputs["msd_out"];
    _msd_config.enabled = true;
    _msd_config.interval = msd_out.value("interval", rbmd::Id{0});
    _msd_config.start_step = msd_out.value("start_step", rbmd::Id{0});
    _msd_config.end_step = msd_out.value("end_step", rbmd::Id{0});
  }

  if (outputs.contains("vacf_out") && outputs["vacf_out"].is_object()) {
    auto& vacf_out = outputs["vacf_out"];
    _vacf_config.enabled = true;
    _vacf_config.interval = vacf_out.value("interval", rbmd::Id{0});
    _vacf_config.start_step = vacf_out.value("start_step", rbmd::Id{0});
    _vacf_config.end_step = vacf_out.value("end_step", rbmd::Id{0});
  }

  if (_msd_config.enabled &&
      (_msd_config.interval <= 0 || _msd_config.end_step < _msd_config.start_step)) {
    Logger::Instance().warn(
        "AnalysisOutput disabled msd_out because interval/start/end are invalid.");
    _msd_config.enabled = false;
  }

  if (_vacf_config.enabled &&
      (_vacf_config.interval <= 0
       || _vacf_config.end_step < _vacf_config.start_step)) {
    Logger::Instance().warn(
        "AnalysisOutput disabled vacf_out because interval/start/end are invalid.");
    _vacf_config.enabled = false;
  }

  if (_rdf_config.enabled) {
    const bool has_group_pairs =
        !_rdf_config.groups.empty() && !_rdf_config.group_pairs.empty();
    const bool has_legacy_pairs = !_rdf_config.atoms_pair.empty();
    const bool rdf_params_valid = _rdf_config.interval > 0
                                  && _rdf_config.radius > rbmd::Real(0)
                                  && _rdf_config.dr > rbmd::Real(0)
                                  && (has_group_pairs || has_legacy_pairs);
    if (!rdf_params_valid) {
      Logger::Instance().warn(
          "AnalysisOutput disabled rdf_out because interval/radius/dr and RDF pair config are invalid.");
      _rdf_config.enabled = false;
    } else {
      _rdf_num_bins = std::max(
          static_cast<size_t>(1),
          static_cast<size_t>(std::ceil(_rdf_config.radius / _rdf_config.dr)));
      _rdf_bin_centers.resize(_rdf_num_bins, 0.0);
      _rdf_shell_volumes.resize(_rdf_num_bins, 0.0);
      for (size_t bin = 0; bin < _rdf_num_bins; ++bin) {
        const double lower = static_cast<double>(bin) * _rdf_config.dr;
        const double upper =
            std::min(static_cast<double>(_rdf_config.radius),
                     lower + static_cast<double>(_rdf_config.dr));
        _rdf_bin_centers[bin] = 0.5 * (lower + upper);
        _rdf_shell_volumes[bin] =
            kFourPiOverThree
            * (upper * upper * upper - lower * lower * lower);
      }

      if (_rdf_config.statistics_rdf_steps > 0 && _rdf_config.interval > 0) {
        _rdf_flush_interval =
            std::max<rbmd::Id>(1, (_rdf_config.statistics_rdf_steps
                                   + _rdf_config.interval - 1)
                                      / _rdf_config.interval);
      }

      _rdf_pairs.clear();
      if (has_group_pairs) {
        ResolveRdfPairsFromGroups(_rdf_config, _rdf_num_bins, _rdf_pairs);
      } else {
        ResolveRdfPairsFromLegacyAtomsPair(_rdf_config, _rdf_num_bins,
                                           _rdf_pairs);
      }

      if (_rdf_pairs.empty()) {
        Logger::Instance().warn(
            "AnalysisOutput disabled rdf_out because no valid RDF pair entries remain.");
        _rdf_config.enabled = false;
      }
    }
  }

  if (!_rdf_config.enabled && !_msd_config.enabled && !_vacf_config.enabled) {
    _initialized = true;
    return;
  }

  auto prepare_output_files = [this]() {
    if (!_should_write_files) {
      return true;
    }

    auto touch_trunc_file = [](const std::string& path) {
      std::ofstream file;
      if (!OpenOutputFile(file, path, std::ios::out | std::ios::trunc)) {
        return false;
      }
      return true;
    };

    if (_msd_config.enabled) {
      if (!OpenOutputFile(_msd_state.file, "msd.txt",
                          std::ios::out | std::ios::trunc)) {
        return false;
      }
      _msd_state.file << "# step delta_time msd shared_atoms reference_step\n";
      _msd_state.file.flush();
    }

    if (_vacf_config.enabled) {
      if (!OpenOutputFile(_vacf_state.file, "vacf.txt",
                          std::ios::out | std::ios::trunc)) {
        return false;
      }
      _vacf_state.file
          << "# step delta_time vacf shared_atoms reference_step\n";
      _vacf_state.file.flush();
    }

    if (_rdf_config.enabled) {
      for (const auto& pair_state : _rdf_pairs) {
        if (!touch_trunc_file(pair_state.path)) {
          return false;
        }
      }
    }

    return true;
  };

  int file_open_ok = prepare_output_files() ? 1 : 0;
#ifdef USE_MPI
  if (_mpi_root_only_mode) {
    MPI_Bcast(&file_open_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
#endif
  if (!file_open_ok) {
    _should_write_files = false;
    _initialized = true;
    return;
  }

  if (_mpi_root_only_mode) {
    if (!_should_write_files) {
      _initialized = true;
      return;
    }
    _output_thread = std::thread(&AnalysisOutput::OutputWorker, this);
    _initialized = true;
    return;
  }

  if (_num_atoms == 0) {
    _initialized = true;
    return;
  }

  _ring_buffer_size = 4;
  CHECK_RUNTIME(STREAM_CREATE(&_stream));
  AllocateRingBuffer(_num_atoms);
  if (_should_write_files) {
    _output_thread = std::thread(&AnalysisOutput::OutputWorker, this);
  }
  _initialized = true;
}

void AnalysisOutput::AllocateRingBuffer(size_t num_atoms) {
  _ring_buffer.resize(_ring_buffer_size);
  for (auto& frame : _ring_buffer) {
    frame.num_atoms = num_atoms;
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_ids),
                             num_atoms * sizeof(rbmd::Id)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_types),
                             num_atoms * sizeof(rbmd::Id)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_ux),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_uy),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_uz),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_vx),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_vy),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_vz),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(NEW_FLAG_EVENT(&frame.copy_complete_event, EVENT_DISABLE));
  }
}

void AnalysisOutput::DeallocateRingBuffer() {
  for (auto& frame : _ring_buffer) {
    if (frame.h_ids) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_ids));
      frame.h_ids = nullptr;
    }
    if (frame.h_types) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_types));
      frame.h_types = nullptr;
    }
    if (frame.h_ux) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_ux));
      frame.h_ux = nullptr;
    }
    if (frame.h_uy) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_uy));
      frame.h_uy = nullptr;
    }
    if (frame.h_uz) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_uz));
      frame.h_uz = nullptr;
    }
    if (frame.h_vx) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_vx));
      frame.h_vx = nullptr;
    }
    if (frame.h_vy) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_vy));
      frame.h_vy = nullptr;
    }
    if (frame.h_vz) {
      CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_vz));
      frame.h_vz = nullptr;
    }
    if (frame.copy_complete_event) {
      CHECK_RUNTIME(EVENT_DESTORY(frame.copy_complete_event));
      frame.copy_complete_event = nullptr;
    }
    frame.num_atoms = 0;
  }
  _ring_buffer.clear();
  _ring_buffer_size = 0;
}

size_t AnalysisOutput::CurrentLocalAtomCount() const {
  if (_structure_info_data && _structure_info_data->_num_atoms &&
      *(_structure_info_data->_num_atoms) > 0) {
    return static_cast<size_t>(*(_structure_info_data->_num_atoms));
  }
  return 0;
}

void AnalysisOutput::CopyNativeAtomsToFrame(
    size_t local_num_atoms, AnalysisFrame& host_frame) const {
  host_frame.ids.resize(local_num_atoms);
  host_frame.types.resize(local_num_atoms);
  host_frame.ux.resize(local_num_atoms);
  host_frame.uy.resize(local_num_atoms);
  host_frame.uz.resize(local_num_atoms);
  host_frame.vx.resize(local_num_atoms);
  host_frame.vy.resize(local_num_atoms);
  host_frame.vz.resize(local_num_atoms);

  if (local_num_atoms == 0) {
    return;
  }

  thrust::copy(_device_data->_d_atoms_id.begin(),
               _device_data->_d_atoms_id.begin() + local_num_atoms,
               host_frame.ids.begin());
  thrust::copy(_device_data->_d_atoms_type.begin(),
               _device_data->_d_atoms_type.begin() + local_num_atoms,
               host_frame.types.begin());
  thrust::copy(_device_data->_d_unwarp_px.begin(),
               _device_data->_d_unwarp_px.begin() + local_num_atoms,
               host_frame.ux.begin());
  thrust::copy(_device_data->_d_unwarp_py.begin(),
               _device_data->_d_unwarp_py.begin() + local_num_atoms,
               host_frame.uy.begin());
  thrust::copy(_device_data->_d_unwarp_pz.begin(),
               _device_data->_d_unwarp_pz.begin() + local_num_atoms,
               host_frame.uz.begin());
  thrust::copy(_device_data->_d_vx.begin(),
               _device_data->_d_vx.begin() + local_num_atoms,
               host_frame.vx.begin());
  thrust::copy(_device_data->_d_vy.begin(),
               _device_data->_d_vy.begin() + local_num_atoms,
               host_frame.vy.begin());
  thrust::copy(_device_data->_d_vz.begin(),
               _device_data->_d_vz.begin() + local_num_atoms,
               host_frame.vz.begin());
}

void AnalysisOutput::Execute(rbmd::Id current_timestep) {
  if (!_initialized) {
    return;
  }

  const rbmd::Id step = current_timestep;

  if (_mpi_root_only_mode) {
    if (!_should_write_files) {
      return;
    }
    ExecuteMpiRootOnly(current_timestep);
    return;
  }

  const AnalysisSchedule schedule = BuildSchedule(step);
  if (!schedule.Any()) {
    return;
  }

  if (_ring_buffer_size == 0) {
    return;
  }

  std::unique_lock<std::mutex> lock(_mutex);
  _cv_not_full.wait(lock, [this] {
    const size_t next_write = (_write_index.load() + 1) % _ring_buffer_size;
    return next_write != _read_index.load() || _stop_flag.load();
  });

  if (_stop_flag.load()) {
    return;
  }

  const size_t write_idx = _write_index.load();
  auto& target_frame = _ring_buffer[write_idx];
  target_frame.timestep = current_timestep;
  target_frame.box_snapshot = *(DataManager::getInstance().getMDData()->_box);
  lock.unlock();

  const size_t real_bytes = _num_atoms * sizeof(rbmd::Real);
  const size_t id_bytes = _num_atoms * sizeof(rbmd::Id);
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_ids,
                             raw_ptr(_device_data->_d_atoms_id), id_bytes, D2H,
                             _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_types,
                             raw_ptr(_device_data->_d_atoms_type), id_bytes,
                             D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_ux,
                             raw_ptr(_device_data->_d_unwarp_px), real_bytes,
                             D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_uy,
                             raw_ptr(_device_data->_d_unwarp_py), real_bytes,
                             D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_uz,
                             raw_ptr(_device_data->_d_unwarp_pz), real_bytes,
                             D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_vx,
                             raw_ptr(_device_data->_d_vx), real_bytes, D2H,
                             _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_vy,
                             raw_ptr(_device_data->_d_vy), real_bytes, D2H,
                             _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_vz,
                             raw_ptr(_device_data->_d_vz), real_bytes, D2H,
                             _stream));
  CHECK_RUNTIME(EVENT_RECORD(target_frame.copy_complete_event, _stream));

  lock.lock();
  _write_index.store((write_idx + 1) % _ring_buffer_size);
  _cv_not_empty.notify_one();
}

void AnalysisOutput::ExecuteMpiRootOnly(rbmd::Id current_timestep) {
#ifdef USE_MPI
  const rbmd::Id step = current_timestep;
  const AnalysisSchedule schedule = BuildSchedule(step);
  if (!schedule.Any()) {
    return;
  }

  const size_t local_num_atoms = CurrentLocalAtomCount();
  const bool is_root = rbmd::mpi::ShouldWriteRootOnlyOutput();
  int local_count_overflow = 0;
  if (local_num_atoms > static_cast<size_t>(std::numeric_limits<int>::max())) {
    local_count_overflow = 1;
  }

  int global_count_overflow = 0;
  MPI_Allreduce(&local_count_overflow, &global_count_overflow, 1, MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
  if (global_count_overflow != 0) {
    if (is_root) {
      Logger::Instance().error(
          "AnalysisOutput MPI gather skipped: local atom count exceeds MPI INT limit.");
    }
    return;
  }

  CopyNativeAtomsToFrame(local_num_atoms, _mpi_local_frame);
  _mpi_local_frame.timestep = current_timestep;

  const auto& local_box = *(DataManager::getInstance().getMDData()->_box);
  _mpi_local_frame.box_snapshot = local_box;
  const std::array<rbmd::Real, 6> local_bounds = {
      local_box._coord_min[0], local_box._coord_min[1], local_box._coord_min[2],
      local_box._coord_max[0], local_box._coord_max[1], local_box._coord_max[2]};

  const int local_count = static_cast<int>(local_num_atoms);

  if (is_root) {
    _mpi_recv_counts.assign(static_cast<size_t>(_mpi_world_size), 0);
    _mpi_recv_displs.assign(static_cast<size_t>(_mpi_world_size), 0);
    _mpi_gathered_box_bounds.resize(static_cast<size_t>(_mpi_world_size)
                                    * local_bounds.size());
  }

  MPI_Gather(&local_count, 1, MPI_INT,
             is_root ? DataOrNull(_mpi_recv_counts) : nullptr, 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(const_cast<rbmd::Real*>(local_bounds.data()),
             static_cast<int>(local_bounds.size()), MPI_RBMD_REAL,
             is_root ? DataOrNull(_mpi_gathered_box_bounds) : nullptr,
             static_cast<int>(local_bounds.size()), MPI_RBMD_REAL, 0,
             MPI_COMM_WORLD);

  if (is_root) {
    int next_displ = 0;
    int displ_overflow = 0;
    for (int rank = 0; rank < _mpi_world_size; ++rank) {
      if (_mpi_recv_counts[static_cast<size_t>(rank)] < 0) {
        displ_overflow = 1;
        break;
      }
      if (_mpi_recv_counts[static_cast<size_t>(rank)] >
          std::numeric_limits<int>::max() - next_displ) {
        displ_overflow = 1;
        break;
      }
      _mpi_recv_displs[static_cast<size_t>(rank)] = next_displ;
      next_displ += _mpi_recv_counts[static_cast<size_t>(rank)];
    }

    MPI_Bcast(&displ_overflow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (displ_overflow != 0) {
      Logger::Instance().error(
          "AnalysisOutput MPI gather skipped: receive displacement exceeds MPI INT limit.");
      return;
    }

    const size_t total_atoms = static_cast<size_t>(next_displ);
    _mpi_root_frame.ids.resize(total_atoms);
    _mpi_root_frame.types.resize(total_atoms);
    _mpi_root_frame.ux.resize(total_atoms);
    _mpi_root_frame.uy.resize(total_atoms);
    _mpi_root_frame.uz.resize(total_atoms);
    _mpi_root_frame.vx.resize(total_atoms);
    _mpi_root_frame.vy.resize(total_atoms);
    _mpi_root_frame.vz.resize(total_atoms);
  } else {
    int displ_overflow = 0;
    MPI_Bcast(&displ_overflow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (displ_overflow != 0) {
      return;
    }
  }

  MPI_Gatherv(DataOrNull(_mpi_local_frame.ids), local_count, MPI_RBMD_ID,
              is_root ? DataOrNull(_mpi_root_frame.ids) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_ID, 0,
              MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.types), local_count, MPI_RBMD_ID,
              is_root ? DataOrNull(_mpi_root_frame.types) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_ID, 0,
              MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.ux), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.ux) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.uy), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.uy) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.uz), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.uz) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.vx), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.vx) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.vy), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.vy) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.vz), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.vz) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);

  if (is_root && _should_write_files) {
    _mpi_root_frame.timestep = current_timestep;
    _mpi_root_frame.box_snapshot =
        BuildGlobalBoxFromLocalBounds(_mpi_gathered_box_bounds, local_box);
    EnqueueFrame(std::move(_mpi_root_frame));
  }
#else
  (void)current_timestep;
#endif
}

void AnalysisOutput::EnqueueFrame(AnalysisFrame&& frame) {
  std::unique_lock<std::mutex> lock(_mutex);
  _cv_not_full.wait(lock, [this] {
    return _pending_frames.size() < _pending_frame_limit || _stop_flag.load();
  });
  if (_stop_flag.load()) {
    return;
  }
  _pending_frames.push_back(std::move(frame));
  lock.unlock();
  _cv_not_empty.notify_one();
}

double AnalysisOutput::StepToTime(rbmd::Id delta_step) const {
  return static_cast<double>(_simulation_timestep)
         * static_cast<double>(delta_step);
}

void AnalysisOutput::CaptureMsdReference(const AnalysisFrame& frame,
                                         rbmd::Id step) {
  if (frame.ids.empty()) {
    return;
  }

  const rbmd::Id max_id =
      *std::max_element(frame.ids.begin(), frame.ids.end());
  if (max_id < 0) {
    return;
  }

  const size_t storage_size = static_cast<size_t>(max_id) + 1;
  _msd_state.present.assign(storage_size, 0);
  _msd_state.ref_ux.assign(storage_size, rbmd::Real(0));
  _msd_state.ref_uy.assign(storage_size, rbmd::Real(0));
  _msd_state.ref_uz.assign(storage_size, rbmd::Real(0));

  for (size_t i = 0; i < frame.ids.size(); ++i) {
    const rbmd::Id atom_id = frame.ids[i];
    if (atom_id < 0) {
      continue;
    }
    const size_t index = static_cast<size_t>(atom_id);
    _msd_state.present[index] = 1;
    _msd_state.ref_ux[index] = frame.ux[i];
    _msd_state.ref_uy[index] = frame.uy[i];
    _msd_state.ref_uz[index] = frame.uz[i];
  }

  _msd_state.reference_step = step;
  _msd_state.reference_ready = true;
  if (_msd_state.file.is_open()) {
    _msd_state.file << step << ' ' << 0.0 << ' ' << 0.0 << ' '
                    << frame.ids.size() << ' ' << step << '\n';
    _msd_state.file.flush();
  }
}

void AnalysisOutput::CaptureVacfReference(const AnalysisFrame& frame,
                                          rbmd::Id step) {
  if (frame.ids.empty()) {
    return;
  }

  const rbmd::Id max_id =
      *std::max_element(frame.ids.begin(), frame.ids.end());
  if (max_id < 0) {
    return;
  }

  const size_t storage_size = static_cast<size_t>(max_id) + 1;
  _vacf_state.present.assign(storage_size, 0);
  _vacf_state.ref_vx.assign(storage_size, rbmd::Real(0));
  _vacf_state.ref_vy.assign(storage_size, rbmd::Real(0));
  _vacf_state.ref_vz.assign(storage_size, rbmd::Real(0));

  double vacf0 = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < frame.ids.size(); ++i) {
    const rbmd::Id atom_id = frame.ids[i];
    if (atom_id < 0) {
      continue;
    }
    const size_t index = static_cast<size_t>(atom_id);
    _vacf_state.present[index] = 1;
    _vacf_state.ref_vx[index] = frame.vx[i];
    _vacf_state.ref_vy[index] = frame.vy[i];
    _vacf_state.ref_vz[index] = frame.vz[i];
    vacf0 += static_cast<double>(frame.vx[i]) * frame.vx[i]
             + static_cast<double>(frame.vy[i]) * frame.vy[i]
             + static_cast<double>(frame.vz[i]) * frame.vz[i];
    ++count;
  }

  _vacf_state.reference_step = step;
  _vacf_state.reference_ready = true;
  if (_vacf_state.file.is_open() && count > 0) {
    _vacf_state.file << step << ' ' << 0.0 << ' ' << (vacf0 / count) << ' '
                     << count << ' ' << step << '\n';
    _vacf_state.file.flush();
  }
}

void AnalysisOutput::UpdateMsd(const AnalysisFrame& frame, rbmd::Id step) {
  if (!_msd_config.enabled) {
    return;
  }

  if (!_msd_state.reference_ready) {
    CaptureMsdReference(frame, step);
    return;
  }

  double sum = 0.0;
  size_t shared_atoms = 0;
  for (size_t i = 0; i < frame.ids.size(); ++i) {
    const rbmd::Id atom_id = frame.ids[i];
    if (atom_id < 0) {
      continue;
    }
    const size_t index = static_cast<size_t>(atom_id);
    if (index >= _msd_state.present.size() || _msd_state.present[index] == 0) {
      continue;
    }

    const double dx = static_cast<double>(frame.ux[i]) - _msd_state.ref_ux[index];
    const double dy = static_cast<double>(frame.uy[i]) - _msd_state.ref_uy[index];
    const double dz = static_cast<double>(frame.uz[i]) - _msd_state.ref_uz[index];
    sum += dx * dx + dy * dy + dz * dz;
    ++shared_atoms;
  }

  if (_msd_state.file.is_open() && shared_atoms > 0) {
    _msd_state.file << step << ' '
                    << StepToTime(step - _msd_state.reference_step) << ' '
                    << (sum / shared_atoms) << ' ' << shared_atoms << ' '
                    << _msd_state.reference_step << '\n';
    _msd_state.file.flush();
  }
}

void AnalysisOutput::UpdateVacf(const AnalysisFrame& frame, rbmd::Id step) {
  if (!_vacf_config.enabled) {
    return;
  }

  if (!_vacf_state.reference_ready) {
    CaptureVacfReference(frame, step);
    return;
  }

  double sum = 0.0;
  size_t shared_atoms = 0;
  for (size_t i = 0; i < frame.ids.size(); ++i) {
    const rbmd::Id atom_id = frame.ids[i];
    if (atom_id < 0) {
      continue;
    }
    const size_t index = static_cast<size_t>(atom_id);
    if (index >= _vacf_state.present.size() || _vacf_state.present[index] == 0) {
      continue;
    }

    sum += static_cast<double>(frame.vx[i]) * _vacf_state.ref_vx[index]
           + static_cast<double>(frame.vy[i]) * _vacf_state.ref_vy[index]
           + static_cast<double>(frame.vz[i]) * _vacf_state.ref_vz[index];
    ++shared_atoms;
  }

  if (_vacf_state.file.is_open() && shared_atoms > 0) {
    _vacf_state.file << step << ' '
                     << StepToTime(step - _vacf_state.reference_step) << ' '
                     << (sum / shared_atoms) << ' ' << shared_atoms << ' '
                     << _vacf_state.reference_step << '\n';
    _vacf_state.file.flush();
  }
}

void AnalysisOutput::WriteRdfFile(const RdfPairState& pair_state) const {
  if (!_should_write_files) {
    return;
  }

  std::ofstream rdf_file;
  if (!OpenOutputFile(rdf_file, pair_state.path,
                      std::ios::out | std::ios::trunc)) {
    return;
  }

  rdf_file << "# lhs_label " << pair_state.lhs_label << '\n';
  rdf_file << "# rhs_label " << pair_state.rhs_label << '\n';
  rdf_file << "# lhs_types";
  for (const auto type_value : pair_state.lhs_types) {
    rdf_file << ' ' << (type_value + 1);
  }
  rdf_file << '\n';
  rdf_file << "# rhs_types";
  for (const auto type_value : pair_state.rhs_types) {
    rdf_file << ' ' << (type_value + 1);
  }
  rdf_file << '\n';
  rdf_file << "# radius " << _rdf_config.radius << '\n';
  rdf_file << "# dr " << _rdf_config.dr << '\n';
  rdf_file << "# sampled_frames " << pair_state.sampled_frames << '\n';
  rdf_file << "# columns: r g_r counts\n";

  for (size_t bin = 0; bin < pair_state.counts.size(); ++bin) {
    double g_r = 0.0;
    if (pair_state.normalization_sum > 0.0 && _rdf_shell_volumes[bin] > 0.0) {
      g_r = pair_state.counts[bin]
            / (pair_state.normalization_sum * _rdf_shell_volumes[bin]);
    }
    rdf_file << _rdf_bin_centers[bin] << ' ' << g_r << ' '
             << pair_state.counts[bin] << '\n';
  }
}

void AnalysisOutput::FlushRdfFiles(bool force) {
  if (!_rdf_config.enabled || !_should_write_files) {
    return;
  }

  for (const auto& pair_state : _rdf_pairs) {
    if (pair_state.sampled_frames == 0) {
      continue;
    }
    if (force || (_rdf_flush_interval > 0
                  && pair_state.sampled_frames % _rdf_flush_interval == 0)) {
      WriteRdfFile(pair_state);
    }
  }
}

void AnalysisOutput::UpdateRdf(const AnalysisFrame& frame, rbmd::Id step) {
  (void)step;

  if (!_rdf_config.enabled || _rdf_pairs.empty() || frame.ids.size() < 2) {
    return;
  }

  const Box& box = frame.box_snapshot;
  if (box._type != Box::BoxType::ORTHOGONAL) {
    Logger::Instance().warn(
        "AnalysisOutput RDF currently supports only orthogonal boxes. This frame is skipped.");
    return;
  }

  const double volume = static_cast<double>(box._length[0]) * box._length[1]
                        * box._length[2];
  if (volume <= 0.0) {
    return;
  }

  const size_t num_atoms = frame.ids.size();
  rbmd::Id max_type = 0;
  for (const auto atom_type : frame.types) {
    max_type = std::max(max_type, atom_type);
  }
  std::vector<size_t> type_counts(static_cast<size_t>(max_type) + 1, 0);
  for (const auto atom_type : frame.types) {
    if (atom_type >= 0) {
      ++type_counts[static_cast<size_t>(atom_type)];
    }
  }

  const size_t type_slots = type_counts.size();
  std::vector<std::vector<size_t>> pair_indices_by_type_pair(type_slots
                                                             * type_slots);
  std::vector<unsigned char> rdf_active_types(type_slots, 0);
  auto add_pair_mapping = [&](rbmd::Id lhs_type, rbmd::Id rhs_type,
                              size_t pair_index) {
    if (lhs_type < 0 || rhs_type < 0) {
      return;
    }
    const size_t lhs_index = static_cast<size_t>(lhs_type);
    const size_t rhs_index = static_cast<size_t>(rhs_type);
    if (lhs_index >= type_slots || rhs_index >= type_slots) {
      return;
    }

    rdf_active_types[lhs_index] = 1;
    rdf_active_types[rhs_index] = 1;
    AddUniqueRdfPairIndex(
        pair_indices_by_type_pair[lhs_index * type_slots + rhs_index],
        pair_index);
  };

  for (size_t pair_idx = 0; pair_idx < _rdf_pairs.size(); ++pair_idx) {
    const auto& pair_state = _rdf_pairs[pair_idx];
    if (pair_state.self_pair) {
      for (const auto lhs_type : pair_state.lhs_types) {
        for (const auto rhs_type : pair_state.lhs_types) {
          add_pair_mapping(lhs_type, rhs_type, pair_idx);
        }
      }
    } else {
      for (const auto lhs_type : pair_state.lhs_types) {
        for (const auto rhs_type : pair_state.rhs_types) {
          add_pair_mapping(lhs_type, rhs_type, pair_idx);
          add_pair_mapping(rhs_type, lhs_type, pair_idx);
        }
      }
    }
  }

  std::vector<rbmd::Real> px(num_atoms, rbmd::Real(0));
  std::vector<rbmd::Real> py(num_atoms, rbmd::Real(0));
  std::vector<rbmd::Real> pz(num_atoms, rbmd::Real(0));
  std::vector<int> cell_ix(num_atoms, 0);
  std::vector<int> cell_iy(num_atoms, 0);
  std::vector<int> cell_iz(num_atoms, 0);

  const int nx = std::max(
      1, static_cast<int>(std::floor(box._length[0] / _rdf_config.radius)));
  const int ny = std::max(
      1, static_cast<int>(std::floor(box._length[1] / _rdf_config.radius)));
  const int nz = std::max(
      1, static_cast<int>(std::floor(box._length[2] / _rdf_config.radius)));
  const size_t total_cells =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  const double cell_x = static_cast<double>(box._length[0]) / nx;
  const double cell_y = static_cast<double>(box._length[1]) / ny;
  const double cell_z = static_cast<double>(box._length[2]) / nz;

  std::vector<int> head(total_cells, -1);
  std::vector<int> next(num_atoms, -1);
  std::vector<size_t> rdf_atom_indices;
  rdf_atom_indices.reserve(num_atoms);

  for (size_t i = 0; i < num_atoms; ++i) {
    const rbmd::Id atom_type = frame.types[i];
    if (atom_type < 0 || static_cast<size_t>(atom_type) >= type_slots
        || rdf_active_types[static_cast<size_t>(atom_type)] == 0) {
      continue;
    }

    px[i] = WrapCoordinateToPrimary(box, frame.ux[i], 0, box._pbc_x);
    py[i] = WrapCoordinateToPrimary(box, frame.uy[i], 1, box._pbc_y);
    pz[i] = WrapCoordinateToPrimary(box, frame.uz[i], 2, box._pbc_z);

    const int ix = std::min(
        nx - 1,
        std::max(0, static_cast<int>((px[i] - box._coord_min[0]) / cell_x)));
    const int iy = std::min(
        ny - 1,
        std::max(0, static_cast<int>((py[i] - box._coord_min[1]) / cell_y)));
    const int iz = std::min(
        nz - 1,
        std::max(0, static_cast<int>((pz[i] - box._coord_min[2]) / cell_z)));

    cell_ix[i] = ix;
    cell_iy[i] = iy;
    cell_iz[i] = iz;

    const size_t cell_index =
        (static_cast<size_t>(iz) * static_cast<size_t>(ny)
         + static_cast<size_t>(iy))
            * static_cast<size_t>(nx)
        + static_cast<size_t>(ix);
    next[i] = head[cell_index];
    head[cell_index] = static_cast<int>(i);
    rdf_atom_indices.push_back(i);
  }

  const size_t worker_threads =
      std::min(_rdf_worker_threads,
               std::max(static_cast<size_t>(1), rdf_atom_indices.size()));
  std::vector<std::vector<std::vector<double>>> local_counts(
      worker_threads,
      std::vector<std::vector<double>>(
          _rdf_pairs.size(), std::vector<double>(_rdf_num_bins, 0.0)));

  auto wrap_cell = [](int coord, int width, bool periodic, bool& valid) {
    if (periodic) {
      coord %= width;
      if (coord < 0) {
        coord += width;
      }
      valid = true;
      return coord;
    }

    valid = coord >= 0 && coord < width;
    return coord;
  };

  const double r_max_sq =
      static_cast<double>(_rdf_config.radius) * _rdf_config.radius;
  const size_t active_atom_count = rdf_atom_indices.size();
  const size_t chunk =
      (active_atom_count + worker_threads - 1) / worker_threads;
  std::vector<std::thread> workers;
  workers.reserve(worker_threads > 1 ? worker_threads - 1 : 0);

  auto worker_body = [&](size_t worker_id, size_t begin, size_t end) {
    auto& worker_hist = local_counts[worker_id];
    for (size_t active_index = begin; active_index < end; ++active_index) {
      const size_t i = rdf_atom_indices[active_index];
      const rbmd::Id lhs_type = frame.types[i];
      if (lhs_type < 0 || static_cast<size_t>(lhs_type) >= type_slots
          || rdf_active_types[static_cast<size_t>(lhs_type)] == 0) {
        continue;
      }

      std::array<int, 27> neighbor_cells{};
      size_t neighbor_count = 0;

      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            bool valid_x = false;
            bool valid_y = false;
            bool valid_z = false;
            const int nix = wrap_cell(cell_ix[i] + dx, nx, box._pbc_x, valid_x);
            const int niy = wrap_cell(cell_iy[i] + dy, ny, box._pbc_y, valid_y);
            const int niz = wrap_cell(cell_iz[i] + dz, nz, box._pbc_z, valid_z);
            if (!valid_x || !valid_y || !valid_z) {
              continue;
            }

            const int cell_id = (niz * ny + niy) * nx + nix;
            bool seen = false;
            for (size_t k = 0; k < neighbor_count; ++k) {
              if (neighbor_cells[k] == cell_id) {
                seen = true;
                break;
              }
            }
            if (!seen) {
              neighbor_cells[neighbor_count++] = cell_id;
            }
          }
        }
      }

      for (size_t k = 0; k < neighbor_count; ++k) {
        for (int j = head[static_cast<size_t>(neighbor_cells[k])]; j != -1;
             j = next[static_cast<size_t>(j)]) {
          if (static_cast<size_t>(j) <= i) {
            continue;
          }

          const rbmd::Id rhs_type = frame.types[static_cast<size_t>(j)];
          if (rhs_type < 0 || static_cast<size_t>(rhs_type) >= type_slots) {
            continue;
          }

          const auto& matching_pair_indices =
              pair_indices_by_type_pair[static_cast<size_t>(lhs_type)
                                            * type_slots
                                        + static_cast<size_t>(rhs_type)];
          if (matching_pair_indices.empty()) {
            continue;
          }

          rbmd::Real dx = frame.ux[static_cast<size_t>(j)] - frame.ux[i];
          rbmd::Real dy = frame.uy[static_cast<size_t>(j)] - frame.uy[i];
          rbmd::Real dz = frame.uz[static_cast<size_t>(j)] - frame.uz[i];
          MinImageDistance_fix(box, dx, dy, dz);

          const double distance_sq = static_cast<double>(dx) * dx
                                     + static_cast<double>(dy) * dy
                                     + static_cast<double>(dz) * dz;
          if (distance_sq >= r_max_sq) {
            continue;
          }

          const double distance = std::sqrt(distance_sq);
          size_t bin = static_cast<size_t>(distance / _rdf_config.dr);
          if (bin >= _rdf_num_bins) {
            bin = _rdf_num_bins - 1;
          }

          for (const size_t pair_idx : matching_pair_indices) {
            worker_hist[pair_idx][bin] += 1.0;
          }
        }
      }
    }
  };

  for (size_t worker_id = 1; worker_id < worker_threads; ++worker_id) {
    const size_t begin = worker_id * chunk;
    const size_t end = std::min(active_atom_count, begin + chunk);
    if (begin >= end) {
      break;
    }
    workers.emplace_back(worker_body, worker_id, begin, end);
  }

  const size_t main_end = std::min(active_atom_count, chunk);
  worker_body(0, 0, main_end);

  for (auto& worker : workers) {
    worker.join();
  }

  for (size_t pair_idx = 0; pair_idx < _rdf_pairs.size(); ++pair_idx) {
    auto& pair_state = _rdf_pairs[pair_idx];
    for (size_t worker_id = 0; worker_id < worker_threads; ++worker_id) {
      for (size_t bin = 0; bin < _rdf_num_bins; ++bin) {
        pair_state.counts[bin] += local_counts[worker_id][pair_idx][bin];
      }
    }

    const size_t group_count_i =
        CountMatchingTypes(pair_state.lhs_types, type_counts);
    const size_t group_count_j =
        CountMatchingTypes(pair_state.rhs_types, type_counts);

    if (pair_state.self_pair) {
      if (group_count_i >= 2) {
        pair_state.normalization_sum +=
            0.5 * static_cast<double>(group_count_i)
            * static_cast<double>(group_count_i - 1) / volume;
      }
    } else if (group_count_i > 0 && group_count_j > 0) {
      pair_state.normalization_sum +=
          static_cast<double>(group_count_i) * group_count_j / volume;
    }

    ++pair_state.sampled_frames;
  }

  FlushRdfFiles(false);
}

void AnalysisOutput::ProcessFrame(const AnalysisFrame& frame) {
  if (!_should_write_files) {
    return;
  }

  const rbmd::Id step = static_cast<rbmd::Id>(frame.timestep);
  const AnalysisSchedule schedule = BuildSchedule(step);
  if (!schedule.Any()) {
    return;
  }

  if (schedule.msd) {
    UpdateMsd(frame, step);
  }
  if (schedule.vacf) {
    UpdateVacf(frame, step);
  }
  if (schedule.rdf) {
    UpdateRdf(frame, step);
  }
}

void AnalysisOutput::OutputWorker() {
  while (true) {
    AnalysisFrame frame;
    bool has_frame = false;
    bool from_ring = false;
    size_t read_idx = 0;

    {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv_not_empty.wait(lock, [this] {
        return !_pending_frames.empty()
               || (_ring_buffer_size > 0
                   && _read_index.load() != _write_index.load())
               || _stop_flag.load();
      });

      if (!_pending_frames.empty()) {
        frame = std::move(_pending_frames.front());
        _pending_frames.pop_front();
        _cv_not_full.notify_one();
        has_frame = true;
      } else if (_ring_buffer_size > 0
                 && _read_index.load() != _write_index.load()) {
        read_idx = _read_index.load();
        from_ring = true;
        has_frame = true;
      } else if (_stop_flag.load()) {
        break;
      }
    }

    if (!has_frame) {
      continue;
    }

    if (from_ring) {
      const auto& source_frame = _ring_buffer[read_idx];
      CHECK_RUNTIME(EVENT_SYNC(source_frame.copy_complete_event));

      frame.timestep = source_frame.timestep;
      frame.box_snapshot = source_frame.box_snapshot;
      frame.ids.assign(source_frame.h_ids,
                       source_frame.h_ids + source_frame.num_atoms);
      frame.types.assign(source_frame.h_types,
                         source_frame.h_types + source_frame.num_atoms);
      frame.ux.assign(source_frame.h_ux,
                      source_frame.h_ux + source_frame.num_atoms);
      frame.uy.assign(source_frame.h_uy,
                      source_frame.h_uy + source_frame.num_atoms);
      frame.uz.assign(source_frame.h_uz,
                      source_frame.h_uz + source_frame.num_atoms);
      frame.vx.assign(source_frame.h_vx,
                      source_frame.h_vx + source_frame.num_atoms);
      frame.vy.assign(source_frame.h_vy,
                      source_frame.h_vy + source_frame.num_atoms);
      frame.vz.assign(source_frame.h_vz,
                      source_frame.h_vz + source_frame.num_atoms);

      std::lock_guard<std::mutex> lock(_mutex);
      _read_index.store((read_idx + 1) % _ring_buffer_size);
      _cv_not_full.notify_one();
    }

    ProcessFrame(frame);
  }

  FlushRdfFiles(true);
  if (_msd_state.file.is_open()) {
    _msd_state.file.close();
  }
  if (_vacf_state.file.is_open()) {
    _vacf_state.file.close();
  }
}
