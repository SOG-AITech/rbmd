#include "TrajectoryOutput.h"
#include "RegularGzipStream.hpp"
#include "SelectedTrajectoryOutput.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <experimental/filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include "Logger.hpp"
#include "common/mpi_root_guard.hpp"
#include "common/rbmd_define.h"
#include "linked_cell_locator.h"
#include "memory_utils.h"
#include "spdlog/fmt/fmt.h"
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
                    std::ios::openmode mode,
                    std::unique_ptr<RegularGzipStream>* compressor = nullptr,
                    bool gzip_output = false,
                    std::uint64_t output_limit = std::numeric_limits<std::uint64_t>::max()) {
  try {
    const auto status = fs::symlink_status(path);
    if (fs::exists(status) && !fs::is_regular_file(status)) {
      Logger::Instance().error(
          "Refuse to open output file '{}' in cwd '{}': target is not a regular file.",
          path, SafeCurrentPath());
      return false;
    }
  } catch (const std::exception&) {
    Logger::Instance().error(
        "Failed to inspect output target '{}' in cwd '{}'.", path,
        SafeCurrentPath());
    return false;
  }

  file.open(path, mode);
  if (!file.is_open()) {
    Logger::Instance().error("Failed to open output file '{}' in cwd '{}'.", path,
                             SafeCurrentPath());
    return false;
  }
  if (gzip_output) {
    if (!compressor) throw std::runtime_error("Missing gzip stream owner");
    try { *compressor = std::make_unique<RegularGzipStream>(file, 1, output_limit); }
    catch (const std::exception& e) { Logger::Instance().error("gzip output init failed: {}", e.what()); file.close(); return false; }
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

void ResizeGatherBuffer(TrajectoryGatherBuffer& buffer, size_t num_atoms) {
  buffer.ids.resize(num_atoms);
  buffer.types.resize(num_atoms);
  buffer.charge.resize(num_atoms);
  buffer.px.resize(num_atoms);
  buffer.py.resize(num_atoms);
  buffer.pz.resize(num_atoms);
  buffer.ux.resize(num_atoms);
  buffer.uy.resize(num_atoms);
  buffer.uz.resize(num_atoms);
  buffer.vx.resize(num_atoms);
  buffer.vy.resize(num_atoms);
  buffer.vz.resize(num_atoms);
}

size_t EstimateOptimalBufferSize(size_t single_frame_memory) {
  size_t ideal_size = single_frame_memory * 2;

  const size_t min_buffer = 2 * 1024 * 1024;   // min 2MB
  const size_t max_buffer = 512 * 1024 * 1024;  // max 512MB

  if (ideal_size < min_buffer) {
    ideal_size = min_buffer;
  } else if (ideal_size > max_buffer) {
    ideal_size = max_buffer;
  }

  size_t aligned_size = 1;
  while (aligned_size < ideal_size) {
    aligned_size <<= 1;
  }

  // 确保对齐后的值仍在范围内
  if (aligned_size > max_buffer) {
    aligned_size = max_buffer;
  }

  return aligned_size;
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
}  // namespace

TrajectoryOutput::TrajectoryOutput(double available_memory_usage_ratio)
    : _memory_usage_ratio(available_memory_usage_ratio),
      _output_thread(),
      _mutex(),
      _cv_not_full(),
      _cv_not_empty(),
      _stream(nullptr),
      _ring_buffer_size(0),
      _num_atoms(0),
      _interval(1),
      _write_index(0),
      _read_index(0),
      _stop_flag(false),
      _initialized(false) {}

// TrajectoryOutput::~TrajectoryOutput() {
//   _stop_flag = true;
//   _cv_not_empty.notify_all();
//   _cv_not_full.notify_all();
//   if (_output_thread.joinable()) {
//     _output_thread.join();
//   }
//   DeallocateRingBuffer();
//   if (_stream) {
//     CHECK_RUNTIME(STREAM_DESTORY(_stream));
//   }
// }

TrajectoryOutput::~TrajectoryOutput() {
  using Clock = std::chrono::steady_clock;
  auto t_all0 = Clock::now();

  _stop_flag = true;

  auto t0 = Clock::now();
  _cv_not_empty.notify_all();
  _cv_not_full.notify_all();
  auto t1 = Clock::now();
  double notify_s = std::chrono::duration<double>(t1 - t0).count();

  double join_s = 0.0;
  if (_output_thread.joinable()) {
    auto tj0 = Clock::now();
    _output_thread.join();
    auto tj1 = Clock::now();
    join_s = std::chrono::duration<double>(tj1 - tj0).count();
  }

  double stream_sync_s = 0.0;
  if (_stream) {
    auto ts0 = Clock::now();
    CHECK_RUNTIME(STREAM_SYNC(_stream));
    auto ts1 = Clock::now();
    stream_sync_s = std::chrono::duration<double>(ts1 - ts0).count();
  }

  auto td0 = Clock::now();
  DeallocateRingBuffer();
  auto td1 = Clock::now();
  double dealloc_s = std::chrono::duration<double>(td1 - td0).count();

  double destroy_s = 0.0;
  if (_stream) {
    auto tx0 = Clock::now();
    CHECK_RUNTIME(STREAM_DESTORY(_stream));
    auto tx1 = Clock::now();
    destroy_s = std::chrono::duration<double>(tx1 - tx0).count();
  }

  if (_mpi_compressor) { _mpi_compressor->Finish(); _mpi_compressor.reset(); }
  if (_mpi_trj_file.is_open()) {
    _mpi_trj_file.close();
  }

  auto t_all1 = Clock::now();
  double total_s = std::chrono::duration<double>(t_all1 - t_all0).count();

  Logger::Instance().info(
      "TrajectoryOutput dtor timing: notify={:.3f} s, join={:.3f} s, stream_sync={:.3f} s, deallocate={:.3f} s, destroy_stream={:.3f} s, total={:.3f} s",
      notify_s, join_s, stream_sync_s, dealloc_s, destroy_s, total_s);
}


void TrajectoryOutput::Init() {
  auto config = DataManager::getInstance().getConfigData();
  if (!config->PathExists({"outputs", "trajectory_out"})) {
    _interval = 0;
    _initialized = true;
    Logger::Instance().info(
        "TrajectoryOutput disabled: outputs.trajectory_out is not configured.");
    return;
  }

  bool enabled = true;
  if (config->PathExists({"outputs", "trajectory_out", "enabled"})) {
    enabled = config->Get<bool>("enabled", "outputs", "trajectory_out");
  }

  if (!enabled) {
    _interval = 0;
    _initialized = true;
    Logger::Instance().info("TrajectoryOutput disabled by configuration.");
    return;
  }

  if (config->PathExists({"outputs", "trajectory_out", "selection"}) ||
      config->PathExists({"outputs", "trajectory_out", "fields"})) {
    _selected_output = std::make_unique<SelectedTrajectoryOutput>();
    _selected_output->Init();
    _initialized = true;
    return;
  }

  _interval = config->Get<rbmd::Id>("interval", "outputs", "trajectory_out");
  _trajectory_first_step = 0;
  _trajectory_last_step = std::numeric_limits<rbmd::Id>::max();
  if (config->PathExists({"outputs", "trajectory_out", "first_step"}))
    _trajectory_first_step = config->Get<rbmd::Id>("first_step", "outputs", "trajectory_out");
  if (config->PathExists({"outputs", "trajectory_out", "last_step"}))
    _trajectory_last_step = config->Get<rbmd::Id>("last_step", "outputs", "trajectory_out");
  if (_trajectory_first_step < 0 || _trajectory_last_step < _trajectory_first_step)
    throw std::runtime_error("Invalid trajectory first_step/last_step");
  _gzip_output = config->PathExists({"outputs", "trajectory_out", "gzip"}) &&
    config->Get<bool>("gzip", "outputs", "trajectory_out");
  _trajectory_byte_limit = std::numeric_limits<std::uint64_t>::max();
  if (config->PathExists({"outputs", "trajectory_out", "max_compressed_bytes"})) {
    const auto limit = config->Get<rbmd::Id>("max_compressed_bytes", "outputs", "trajectory_out");
    if (limit < 64) throw std::runtime_error("trajectory max_compressed_bytes too small");
    _trajectory_byte_limit = static_cast<std::uint64_t>(limit);
  }
  _num_atoms = CurrentLocalAtomCount();

#ifdef USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
      MPI_Comm_size(MPI_COMM_WORLD, &_mpi_world_size);
      _mpi_root_only_mode = _mpi_world_size > 1;
    }
  }

  if (_mpi_root_only_mode) {
    _szz_only = config->PathExists({"outputs", "trajectory_out", "szz_only"}) &&
        config->Get<bool>("szz_only", "outputs", "trajectory_out");
    if (_szz_only) {
      _szz_final_step = config->Get<rbmd::Id>("num_steps", "execution");
      if (_interval <= 0 || _szz_final_step % _interval != 0)
        throw std::runtime_error("Szz final step must be a multiple of interval");
      int ok = 1;
      if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
        ok = OpenOutputFile(_szz_file, "szz.csv", std::ios::out | std::ios::trunc);
        _szz_file << "step,qnorm,lx,ly,lz";
        for (int n=1; n<=4; ++n) for (char a : {'x','y','z'}) _szz_file << ',' << a << n;
        _szz_file << '\n' << std::setprecision(17);
      }
      MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (!ok) throw std::runtime_error("Cannot open Szz output");
      _initialized = true;
      return;
    }
    rbmd::GlobalCount local_count = static_cast<rbmd::GlobalCount>(_num_atoms);
    rbmd::GlobalCount global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    _mpi_sharded_mode = global_count > std::numeric_limits<int>::max();
    if (config->PathExists({"outputs", "trajectory_out", "mpi_sharded"}))
      _mpi_sharded_mode = _mpi_sharded_mode || config->Get<bool>("mpi_sharded", "outputs", "trajectory_out");
    if (_mpi_sharded_mode) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      int ok = OpenOutputFile(_mpi_trj_file,
          "rbmd.rank" + std::to_string(rank) + ".trj", std::ios::out | std::ios::trunc, &_mpi_compressor, _gzip_output, _trajectory_byte_limit);
      int all_ok = 0;
      MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (!all_ok) throw std::runtime_error("Could not open trajectory shard on all ranks");
      _initialized = true;
      return;
    }
    int file_open_ok = 1;
    if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
      file_open_ok = OpenOutputFile(_mpi_trj_file, "rbmd.trj",
                                    std::ios::out | std::ios::trunc, &_mpi_compressor, _gzip_output, _trajectory_byte_limit)
                         ? 1
                         : 0;
    }
    MPI_Bcast(&file_open_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!file_open_ok) {
      return;
    }

    Logger::Instance().info(
        "TrajectoryOutput enabled in MPI root-only mode across {} ranks.",
        _mpi_world_size);
    _initialized = true;
    return;
  }
#endif

  if (config->PathExists({"outputs", "trajectory_out", "szz_only"}) &&
      config->Get<bool>("szz_only", "outputs", "trajectory_out"))
    throw std::runtime_error("Szz scalar output currently requires multiple MPI ranks");
  if (_num_atoms == 0) {
    Logger::Instance().warn(
        "TrajectoryOutput::Init called with 0 atoms. Output will be disabled.");
    return;
  }

  size_t frame_mem = CalculateSingleFrameMemory(_num_atoms);
  size_t available_mem = rbmd::utils::get_available_memory();
  size_t memory_to_use =
      static_cast<size_t>(available_mem * _memory_usage_ratio);
  size_t calculated_size =
      (memory_to_use > frame_mem) ? (memory_to_use / frame_mem) : 1;

  _ring_buffer_size =
      std::max(static_cast<size_t>(2),
               std::min(calculated_size, static_cast<size_t>(64)));

  Logger::Instance().info("Trajectory Ring Buffer Initializing...");
  Logger::Instance().info("  Available Memory: {:.2f} GB",
                          available_mem / (1024.0 * 1024.0 * 1024.0));
  Logger::Instance().info("  Single Frame Memory: {:.2f} MB",
                          frame_mem / (1024.0 * 1024.0));
  Logger::Instance().info("  Calculated Ring Buffer Size: {}",
                          _ring_buffer_size);

  CHECK_RUNTIME(STREAM_CREATE(&_stream));
  AllocateRingBuffer(_num_atoms);
  _output_thread = std::thread(&TrajectoryOutput::OutputWorker, this);
  _initialized = true;
}

void TrajectoryOutput::AllocateRingBuffer(size_t num_atoms) {
  _ring_buffer.resize(_ring_buffer_size);
  for (auto& frame : _ring_buffer) {
    frame.num_atoms = num_atoms;
    frame.h_px = nullptr;
    frame.h_py = nullptr;
    frame.h_pz = nullptr;
    frame.h_ux = nullptr;
    frame.h_uy = nullptr;
    frame.h_uz = nullptr;
    frame.h_vx = nullptr;
    frame.h_vy = nullptr;
    frame.h_vz = nullptr;
    frame.h_atoms_type = nullptr;
    frame.copy_complete_event = nullptr;
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_px),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_py),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_pz),
                             num_atoms * sizeof(rbmd::Real)));
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
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_atoms_type),
                             num_atoms * sizeof(rbmd::Id)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_charge),
                             num_atoms * sizeof(rbmd::Real)));
    CHECK_RUNTIME(MALLOCHOST(reinterpret_cast<void**>(&frame.h_atoms_id),
                             num_atoms * sizeof(rbmd::Id)));
    CHECK_RUNTIME(NEW_FLAG_EVENT(&frame.copy_complete_event, EVENT_DISABLE));
  }
}

void TrajectoryOutput::DeallocateRingBuffer() {
  for (auto& frame : _ring_buffer) {
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_px));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_py));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_pz));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_ux));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_uy));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_uz));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_vx));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_vy));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_vz));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_atoms_type));
    CHECK_RUNTIME(EVENT_DESTORY(frame.copy_complete_event));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_charge));
    CHECK_RUNTIME(FREE_PINNED_HOST(frame.h_atoms_id));
  }
  _ring_buffer.clear();
}

size_t TrajectoryOutput::CalculateSingleFrameMemory(size_t num_atoms) const {
  size_t mem = 0;
  mem += num_atoms * sizeof(rbmd::Real) * 3;  // px, py, pz
  mem += num_atoms * sizeof(rbmd::Real) * 3;  // pux, puy, puz
  mem += num_atoms * sizeof(rbmd::Real) * 3;  // vx, vy, vz
  mem += num_atoms * sizeof(rbmd::Real);      // charge
  mem += num_atoms * sizeof(rbmd::Id);        // atoms_type
  mem += num_atoms * sizeof(rbmd::Id);        // atom_id_to_idx
  return mem;
}

size_t TrajectoryOutput::CurrentLocalAtomCount() const {
  if (_structure_info_data && _structure_info_data->_num_atoms &&
      *(_structure_info_data->_num_atoms) > 0) {
    return static_cast<size_t>(*(_structure_info_data->_num_atoms));
  }
  return 0;
}

void TrajectoryOutput::CopyNativeAtomsToHost(
    size_t local_num_atoms, TrajectoryGatherBuffer& host_buffer) const {
  ResizeGatherBuffer(host_buffer, local_num_atoms);
  if (local_num_atoms == 0) {
    return;
  }

  thrust::copy(_device_data->_d_atoms_id.begin(),
               _device_data->_d_atoms_id.begin() + local_num_atoms,
               host_buffer.ids.begin());
  thrust::copy(_device_data->_d_atoms_type.begin(),
               _device_data->_d_atoms_type.begin() + local_num_atoms,
               host_buffer.types.begin());
  thrust::copy(_device_data->_d_charge.begin(),
               _device_data->_d_charge.begin() + local_num_atoms,
               host_buffer.charge.begin());
  thrust::copy(_device_data->_d_px.begin(),
               _device_data->_d_px.begin() + local_num_atoms,
               host_buffer.px.begin());
  thrust::copy(_device_data->_d_py.begin(),
               _device_data->_d_py.begin() + local_num_atoms,
               host_buffer.py.begin());
  thrust::copy(_device_data->_d_pz.begin(),
               _device_data->_d_pz.begin() + local_num_atoms,
               host_buffer.pz.begin());
  thrust::copy(_device_data->_d_unwarp_px.begin(),
               _device_data->_d_unwarp_px.begin() + local_num_atoms,
               host_buffer.ux.begin());
  thrust::copy(_device_data->_d_unwarp_py.begin(),
               _device_data->_d_unwarp_py.begin() + local_num_atoms,
               host_buffer.uy.begin());
  thrust::copy(_device_data->_d_unwarp_pz.begin(),
               _device_data->_d_unwarp_pz.begin() + local_num_atoms,
               host_buffer.uz.begin());
  thrust::copy(_device_data->_d_vx.begin(),
               _device_data->_d_vx.begin() + local_num_atoms,
               host_buffer.vx.begin());
  thrust::copy(_device_data->_d_vy.begin(),
               _device_data->_d_vy.begin() + local_num_atoms,
               host_buffer.vy.begin());
  thrust::copy(_device_data->_d_vz.begin(),
               _device_data->_d_vz.begin() + local_num_atoms,
               host_buffer.vz.begin());
}

void TrajectoryOutput::WriteGatheredFrame(
    std::ofstream& trj_file, rbmd::Id timestep, const Box& box,
    const TrajectoryGatherBuffer& host_buffer) const {
  const size_t num_atoms = host_buffer.ids.size();
  const size_t buffer_size =
      EstimateOptimalBufferSize(CalculateSingleFrameMemory(num_atoms));
  std::unique_ptr<char[]> write_buffer(new char[buffer_size]);
  size_t buffer_pos = 0;

  auto flush_buffer = [&]() {
    if (buffer_pos > 0) {
      trj_file.write(write_buffer.get(), buffer_pos);
      buffer_pos = 0;
    }
  };

  auto write_to_buffer = [&](const std::string& str) {
    const char* data = str.c_str();
    const size_t len = str.length();

    if (len > buffer_size) {
      flush_buffer();
      trj_file.write(data, len);
      return;
    }

    if (buffer_pos + len > buffer_size) {
      flush_buffer();
    }

    std::memcpy(write_buffer.get() + buffer_pos, data, len);
    buffer_pos += len;
  };

  write_to_buffer(fmt::format("ITEM: TIMESTEP\n{}\n", timestep));
  write_to_buffer(fmt::format("ITEM: NUMBER OF ATOMS\n{}\n", num_atoms));
  write_to_buffer("ITEM: BOX BOUNDS pp pp pp\n");
  write_to_buffer(
      fmt::format("{} {}\n{} {}\n{} {}\n", box._coord_min[0], box._coord_max[0],
                  box._coord_min[1], box._coord_max[1], box._coord_min[2],
                  box._coord_max[2]));
  write_to_buffer("ITEM: ATOMS id type q x y z xu yu zu vx vy vz\n");

  std::vector<size_t> atom_order(num_atoms);
  std::iota(atom_order.begin(), atom_order.end(), 0);
  std::sort(atom_order.begin(), atom_order.end(),
            [&](size_t lhs, size_t rhs) {
              return host_buffer.ids[lhs] < host_buffer.ids[rhs];
            });

  const size_t atoms_per_batch = 1000;
  std::string batch_buffer;
  batch_buffer.reserve(atoms_per_batch * 200);

  for (size_t sorted_pos = 0; sorted_pos < num_atoms; ++sorted_pos) {
    const size_t idx = atom_order[sorted_pos];
    batch_buffer += fmt::format(
        "{} {} {} {} {} {} {} {} {} {} {} {}\n", host_buffer.ids[idx] + 1,
        host_buffer.types[idx] + 1, host_buffer.charge[idx],
        host_buffer.px[idx], host_buffer.py[idx], host_buffer.pz[idx],
        host_buffer.ux[idx], host_buffer.uy[idx], host_buffer.uz[idx],
        host_buffer.vx[idx], host_buffer.vy[idx], host_buffer.vz[idx]);

    if ((sorted_pos + 1) % atoms_per_batch == 0 ||
        sorted_pos == num_atoms - 1) {
      write_to_buffer(batch_buffer);
      batch_buffer.clear();
    }
  }

  flush_buffer();
}

void TrajectoryOutput::ExecuteMpiRootOnly(rbmd::Id current_timestep) {
#ifdef USE_MPI
  const size_t local_num_atoms = CurrentLocalAtomCount();
  CopyNativeAtomsToHost(local_num_atoms, _mpi_local_frame);

  const auto& local_box = *(DataManager::getInstance().getMDData()->_box);
  const std::array<rbmd::Real, 6> local_bounds = {
      local_box._coord_min[0], local_box._coord_min[1], local_box._coord_min[2],
      local_box._coord_max[0], local_box._coord_max[1], local_box._coord_max[2]};

  if (_szz_only) {
    rbmd::Real low[3], high[3];
    MPI_Allreduce(local_bounds.data(), low, 3, MPI_RBMD_REAL, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_bounds.data()+3, high, 3, MPI_RBMD_REAL, MPI_MAX, MPI_COMM_WORLD);
    double sums[25] = {}, total[25] = {};
    const std::vector<rbmd::Real>* coords[3] = {&_mpi_local_frame.px, &_mpi_local_frame.py, &_mpi_local_frame.pz};
    for (size_t i=0; i<local_num_atoms; ++i) {
      const double q = _mpi_local_frame.charge[i];
      sums[24] += q*q;
      for (int a=0; a<3; ++a) {
        const double phase = 2*std::acos(-1.0)*(double((*coords[a])[i])-low[a])/(double(high[a])-low[a]);
        const double c=std::cos(phase), s=std::sin(phase);
        double cn=c, sn=s;
        for (int n=0; n<4; ++n) {
          const int j=2*(3*n+a);
          sums[j] += q*cn; sums[j+1] += q*sn;
          const double next=cn*c-sn*s; sn=sn*c+cn*s; cn=next;
        }
      }
    }
    MPI_Allreduce(sums, total, 25, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (!(total[24]>0)) throw std::runtime_error("Szz requires nonzero charge norm");
    if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
      _szz_file << current_timestep << ',' << total[24];
      for (int a=0; a<3; ++a) _szz_file << ',' << double(high[a])-low[a];
      for (int j=0; j<24; j+=2) _szz_file << ',' << (total[j]*total[j]+total[j+1]*total[j+1])/total[24];
      _szz_file << '\n'; _szz_file.flush();
      if (!_szz_file.good()) throw std::runtime_error("Szz write failed");
    }
    // Preserve only the final coordinates and velocities for a future continuation.
    if (current_timestep == _szz_final_step) {
      int rank=0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      std::ofstream final_frame;
      int ok=OpenOutputFile(final_frame, "final.rank"+std::to_string(rank)+".trj", std::ios::out|std::ios::trunc), all_ok=0;
      MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
      if (!all_ok) throw std::runtime_error("Cannot write final Szz snapshot");
      std::vector<rbmd::Real> bounds{low[0],low[1],low[2],high[0],high[1],high[2]};
      WriteGatheredFrame(final_frame,current_timestep,BuildGlobalBoxFromLocalBounds(bounds,local_box),_mpi_local_frame);
      final_frame.flush();
      if (!final_frame.good()) throw std::runtime_error("Final Szz snapshot write failed");
    }
    return;
  }
  if (_mpi_sharded_mode) {
    rbmd::Real low[3], high[3];
    MPI_Allreduce(local_bounds.data(), low, 3, MPI_RBMD_REAL, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_bounds.data() + 3, high, 3, MPI_RBMD_REAL, MPI_MAX, MPI_COMM_WORLD);
    std::vector<rbmd::Real> bounds{low[0], low[1], low[2], high[0], high[1], high[2]};
    const Box global_box = BuildGlobalBoxFromLocalBounds(bounds, local_box);
    WriteGatheredFrame(_mpi_trj_file, current_timestep, global_box, _mpi_local_frame);
    return;
  }

  const bool is_root = rbmd::mpi::ShouldWriteRootOnlyOutput();
  const int local_count = static_cast<int>(local_num_atoms);

  if (is_root) {
    _mpi_recv_counts.assign(static_cast<size_t>(_mpi_world_size), 0);
    _mpi_recv_displs.assign(static_cast<size_t>(_mpi_world_size), 0);
    _mpi_gathered_box_bounds.resize(static_cast<size_t>(_mpi_world_size) *
                                    local_bounds.size());
  }

  MPI_Gather(&local_count, 1, MPI_INT, is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
             1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(const_cast<rbmd::Real*>(local_bounds.data()),
             static_cast<int>(local_bounds.size()), MPI_RBMD_REAL,
             is_root ? DataOrNull(_mpi_gathered_box_bounds) : nullptr,
             static_cast<int>(local_bounds.size()), MPI_RBMD_REAL, 0,
             MPI_COMM_WORLD);

  if (is_root) {
    int next_displ = 0;
    for (int rank = 0; rank < _mpi_world_size; ++rank) {
      _mpi_recv_displs[static_cast<size_t>(rank)] = next_displ;
      next_displ += _mpi_recv_counts[static_cast<size_t>(rank)];
    }
    ResizeGatherBuffer(_mpi_root_frame, static_cast<size_t>(next_displ));
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
  MPI_Gatherv(DataOrNull(_mpi_local_frame.charge), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.charge) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.px), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.px) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.py), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.py) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(DataOrNull(_mpi_local_frame.pz), local_count, MPI_RBMD_REAL,
              is_root ? DataOrNull(_mpi_root_frame.pz) : nullptr,
              is_root ? DataOrNull(_mpi_recv_counts) : nullptr,
              is_root ? DataOrNull(_mpi_recv_displs) : nullptr, MPI_RBMD_REAL,
              0, MPI_COMM_WORLD);
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

  if (is_root) {
    const Box global_box =
        BuildGlobalBoxFromLocalBounds(_mpi_gathered_box_bounds, local_box);
    WriteGatheredFrame(_mpi_trj_file, current_timestep, global_box,
                       _mpi_root_frame);
  }
#else
  (void)current_timestep;
#endif
}

void TrajectoryOutput::Execute(rbmd::Id current_timestep) {
  if (_selected_output) {
    _selected_output->Execute(current_timestep);
    return;
  }
  if (!_initialized || _interval <= 0 || current_timestep % _interval != 0) {
    return;
  }

  if (current_timestep != 0 && (current_timestep < _trajectory_first_step ||
      current_timestep > _trajectory_last_step)) return;

  if (_mpi_root_only_mode) {
    ExecuteMpiRootOnly(current_timestep);
    return;
  }

  if (_ring_buffer_size == 0) return;

  std::unique_lock<std::mutex> lock(_mutex);
  _cv_not_full.wait(lock, [this] {
    size_t next_write = (_write_index.load() + 1) % _ring_buffer_size;
    return next_write != _read_index.load();
  });

  size_t write_idx = _write_index.load();
  auto& target_frame = _ring_buffer[write_idx];
  target_frame.timestep = current_timestep;
  target_frame.box_snapshot = *(DataManager::getInstance().getMDData()->_box);

  // Release the lock before the asynchronous copy
  lock.unlock();

  // Asynchronous copy: Device to Host (Pinned Memory)
  const size_t pos_bytes = _num_atoms * sizeof(rbmd::Real);
  const size_t type_bytes = _num_atoms * sizeof(rbmd::Id);
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_px, raw_ptr(_device_data->_d_px),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_py, raw_ptr(_device_data->_d_py),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_pz, raw_ptr(_device_data->_d_pz),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_ux, raw_ptr(_device_data->_d_unwarp_px),
                           pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_uy, raw_ptr(_device_data->_d_unwarp_py),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_uz, raw_ptr(_device_data->_d_unwarp_pz),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_vx, raw_ptr(_device_data->_d_vx),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_vy, raw_ptr(_device_data->_d_vy),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_vz, raw_ptr(_device_data->_d_vz),
                             pos_bytes, D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_atoms_type,
                             raw_ptr(_device_data->_d_atoms_type), type_bytes,
                             D2H, _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(target_frame.h_charge,
                             raw_ptr(_device_data->_d_charge), pos_bytes, D2H,
                             _stream));
  CHECK_RUNTIME(MEMCPY_ASYNC(
      target_frame.h_atoms_id,
      raw_ptr(_device_data->_d_atoms_id),
      type_bytes, D2H, _stream));

  // Record an event in the stream
  CHECK_RUNTIME(EVENT_RECORD(target_frame.copy_complete_event, _stream));

  lock.lock();  // Reacquire the lock to update the index
  _write_index.store((write_idx + 1) % _ring_buffer_size);
  _cv_not_empty.notify_one();
}

void TrajectoryOutput::OutputWorker() {
  std::ofstream trj_file;
  std::unique_ptr<RegularGzipStream> compressor;
  if (!OpenOutputFile(trj_file, "rbmd.trj", std::ios::out | std::ios::trunc, &compressor, _gzip_output, _trajectory_byte_limit)) {
    return;
  }
  const size_t buffer_size = EstimateOptimalBufferSize(CalculateSingleFrameMemory(_num_atoms));
  std::unique_ptr<char[]> write_buffer(new char[buffer_size]);
  size_t buffer_pos = 0;

  // Lambda function to flush buffer
  auto flush_buffer = [&]() {
    if (buffer_pos > 0) {
      trj_file.write(write_buffer.get(), buffer_pos);
      buffer_pos = 0;
    }
  };

  //  Lambda function to write string to buffer
  auto write_to_buffer = [&](const std::string& str) {
    const char* data = str.c_str();
    size_t len = str.length();

    // If single string exceeds buffer size, write directly to file
    if (len > buffer_size) {
      flush_buffer();
      trj_file.write(data, len);
      return;
    }

    // If current data would cause buffer overflow, flush buffer first
    if (buffer_pos + len > buffer_size) {
      flush_buffer();
    }

    // Write to buffer
    std::memcpy(write_buffer.get() + buffer_pos, data, len);
    buffer_pos += len;
  };

  while (true) {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv_not_empty.wait(lock, [this] {
      return (_read_index.load() != _write_index.load()) || _stop_flag.load();
    });

    if (_stop_flag.load() && (_read_index.load() == _write_index.load())) {
      break;
    }

    size_t read_idx = _read_index.load();
    const auto& source_frame = _ring_buffer[read_idx];
    lock.unlock();

    // Wait for the asynchronous copy of this frame to complete
    CHECK_RUNTIME(EVENT_SYNC(source_frame.copy_complete_event));
    // Use fmt::format for all output formatting
    write_to_buffer(fmt::format("ITEM: TIMESTEP\n{}\n", source_frame.timestep));

    write_to_buffer(fmt::format("ITEM: NUMBER OF ATOMS\n{}\n", source_frame.num_atoms));

    write_to_buffer("ITEM: BOX BOUNDS pp pp pp\n");

    // Format box bounds using fmt
    write_to_buffer(fmt::format("{} {}\n{} {}\n{} {}\n",
                                source_frame.box_snapshot._coord_min[0],
                                source_frame.box_snapshot._coord_max[0],
                                source_frame.box_snapshot._coord_min[1],
                                source_frame.box_snapshot._coord_max[1],
                                source_frame.box_snapshot._coord_min[2],
                                source_frame.box_snapshot._coord_max[2]));

    write_to_buffer("ITEM: ATOMS id type q x y z xu yu zu vx vy vz\n");

    // Batch process atom data using fmt
    const size_t atoms_per_batch = 1000;
    std::string batch_buffer;
    batch_buffer.reserve(atoms_per_batch * 200);

    for (size_t i = 0; i < source_frame.num_atoms; ++i) {
      const auto idx = i;

      // Use fmt::format for atom data - much cleaner than snprintf
      batch_buffer += fmt::format("{} {} {} {} {} {} {} {} {} {} {} {}\n",
                                  source_frame.h_atoms_id[i] + 1,
                                  source_frame.h_atoms_type[idx] + 1,
                                  source_frame.h_charge[idx],
                                  source_frame.h_px[idx],
                                  source_frame.h_py[idx],
                                  source_frame.h_pz[idx],
                                  source_frame.h_ux[idx],
                                  source_frame.h_uy[idx],
                                  source_frame.h_uz[idx],
                                  source_frame.h_vx[idx],
                                  source_frame.h_vy[idx],
                                  source_frame.h_vz[idx]);

      // Write batch when full or at the end
      if ((i + 1) % atoms_per_batch == 0 || i == source_frame.num_atoms - 1) {
        write_to_buffer(batch_buffer);
        batch_buffer.clear();
      }
    }


    // Disabling it yields higher performance, but risks losing frame data on
    // crashes
    // flush_buffer();

    lock.lock();
    _read_index.store((read_idx + 1) % _ring_buffer_size);
    _cv_not_full.notify_one();
  }

  // Finally flush the buffer
  flush_buffer();
  if (compressor) { compressor->Finish(); compressor.reset(); }
  if (!trj_file.good()) throw std::runtime_error("Trajectory stream write failed");
  trj_file.close();
}
