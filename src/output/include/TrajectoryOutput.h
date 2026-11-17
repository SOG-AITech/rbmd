#pragma once
#include "RegularGzipStream.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "output.h"
#include "rbmd_define.h"

// A wrapper for device pointers to clarify intent
struct DeviceDataPointers {
  const rbmd::Real* d_px{};
  const rbmd::Real* d_py{};
  const rbmd::Real* d_pz{};
  const rbmd::Real* d_vx{};
  const rbmd::Real* d_vy{};
  const rbmd::Real* d_vz{};
  const rbmd::Id* d_atoms_type{};
};

// Internal frame data, stored as a raw pointer to fixed memory
struct TrajectoryFrame {
  rbmd::Real* h_px{};
  rbmd::Real* h_py{};
  rbmd::Real* h_pz{};
  rbmd::Real* h_ux{};
  rbmd::Real* h_uy{};
  rbmd::Real* h_uz{};
  rbmd::Real* h_vx{};
  rbmd::Real* h_vy{};
  rbmd::Real* h_vz{};
  rbmd::Id* h_atoms_type{};
  rbmd::Real* h_charge{};
  rbmd::Id* h_atoms_id{};
  rbmd::Id timestep{};
  size_t num_atoms{};
  Box box_snapshot;
  EVENT_T copy_complete_event{};
};

struct TrajectoryGatherBuffer {
  std::vector<rbmd::Id> ids{};
  std::vector<rbmd::Id> types{};
  std::vector<rbmd::Real> charge{};
  std::vector<rbmd::Real> px{};
  std::vector<rbmd::Real> py{};
  std::vector<rbmd::Real> pz{};
  std::vector<rbmd::Real> ux{};
  std::vector<rbmd::Real> uy{};
  std::vector<rbmd::Real> uz{};
  std::vector<rbmd::Real> vx{};
  std::vector<rbmd::Real> vy{};
  std::vector<rbmd::Real> vz{};
};

class TrajectoryOutput:public Output {
public:
  // Constructor parameter: the desired ratio of available memory to use
  explicit TrajectoryOutput(double available_memory_usage_ratio = 0.25);
  ~TrajectoryOutput() override;
  // Disallow copying
  TrajectoryOutput(const TrajectoryOutput&) = delete;
  TrajectoryOutput& operator=(const TrajectoryOutput&) = delete;
  TrajectoryOutput(TrajectoryOutput&&) = delete;
  TrajectoryOutput& operator=(TrajectoryOutput&&) = delete;

  // Initializes the object: calculates buffer size, allocates resources, and starts the I/O thread
  void Init() override;

  // Main execution function, called by the main simulation loop.
  void Execute(rbmd::Id current_timestep) override;

private:
  std::unique_ptr<Output> _selected_output;
  void OutputWorker();
  void ExecuteMpiRootOnly(rbmd::Id current_timestep);
  void CopyNativeAtomsToHost(size_t local_num_atoms,
                             TrajectoryGatherBuffer& host_buffer) const;
  void WriteGatheredFrame(std::ofstream& trj_file, rbmd::Id timestep,
                          const Box& box,
                          const TrajectoryGatherBuffer& host_buffer) const;
  void AllocateRingBuffer(size_t num_atoms);
  void DeallocateRingBuffer();
  [[nodiscard]]  size_t CalculateSingleFrameMemory(size_t num_atoms) const;
  [[nodiscard]]  size_t CurrentLocalAtomCount() const;
  [[nodiscard]]     bool _initialized = false;

  // A ring buffer.
  std::vector<TrajectoryFrame> _ring_buffer;
  size_t _ring_buffer_size{0};

  // Index for the blocking queue.
  std::atomic<size_t> _write_index{0};
  std::atomic<size_t> _read_index{0};

  // Synchronization primitive.
  std::thread _output_thread;
  std::mutex _mutex;
  std::condition_variable _cv_not_full;
  std::condition_variable _cv_not_empty;
  std::atomic<bool> _stop_flag{false};

  STREAM _stream;

  // Configuration.
  double _memory_usage_ratio;
  rbmd::Id _interval{};
  rbmd::Id _trajectory_first_step{0};
  rbmd::Id _trajectory_last_step{0};
  bool _gzip_output{false};
  std::uint64_t _trajectory_byte_limit{0};
  std::unique_ptr<RegularGzipStream> _mpi_compressor;
  size_t _num_atoms{0};

  // MPI root-only trajectory output keeps MPI off the background thread.
  bool _mpi_root_only_mode{false};
  bool _mpi_sharded_mode{false};
  bool _szz_only{false};
  rbmd::Id _szz_final_step{0};
  std::ofstream _szz_file;
  int _mpi_world_size{1};
  TrajectoryGatherBuffer _mpi_local_frame;
  TrajectoryGatherBuffer _mpi_root_frame;
  std::vector<int> _mpi_recv_counts;
  std::vector<int> _mpi_recv_displs;
  std::vector<rbmd::Real> _mpi_gathered_box_bounds;
  std::ofstream _mpi_trj_file;
};
