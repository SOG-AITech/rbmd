#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "analysis_types.h"
#include "output.h"

class AnalysisOutput : public Output {
 public:
  AnalysisOutput() = default;
  ~AnalysisOutput() override;

  AnalysisOutput(const AnalysisOutput&) = delete;
  AnalysisOutput& operator=(const AnalysisOutput&) = delete;
  AnalysisOutput(AnalysisOutput&&) = delete;
  AnalysisOutput& operator=(AnalysisOutput&&) = delete;

 void Init() override;
 void Execute(rbmd::Id current_timestep) override;

 public:
  struct AnalysisSchedule {
    bool rdf{false};
    bool msd{false};
    bool vacf{false};

    [[nodiscard]] bool Any() const { return rdf || msd || vacf; }
  };

  struct DirectSeriesState {
    bool reference_ready{false};
    rbmd::Id reference_step{0};
    std::vector<unsigned char> present{};
    std::ofstream file{};
  };

  struct MsdDirectState : DirectSeriesState {
    std::vector<rbmd::Real> ref_ux{};
    std::vector<rbmd::Real> ref_uy{};
    std::vector<rbmd::Real> ref_uz{};
  };

  struct VacfDirectState : DirectSeriesState {
    std::vector<rbmd::Real> ref_vx{};
    std::vector<rbmd::Real> ref_vy{};
    std::vector<rbmd::Real> ref_vz{};
  };

  struct RdfPairState {
    std::string lhs_label{};
    std::string rhs_label{};
    std::vector<rbmd::Id> lhs_types{};
    std::vector<rbmd::Id> rhs_types{};
    bool self_pair{false};
    std::string path{};
    std::vector<double> counts{};
    double normalization_sum{0.0};
    size_t sampled_frames{0};
  };

 private:
  void ExecuteMpiRootOnly(rbmd::Id current_timestep);
  void OutputWorker();
  void EnqueueFrame(AnalysisFrame&& frame);
  void ProcessFrame(const AnalysisFrame& frame);
  void AllocateRingBuffer(size_t num_atoms);
  void DeallocateRingBuffer();
  void CopyNativeAtomsToFrame(size_t local_num_atoms,
                              AnalysisFrame& host_frame) const;
  void CaptureMsdReference(const AnalysisFrame& frame, rbmd::Id step);
  void CaptureVacfReference(const AnalysisFrame& frame, rbmd::Id step);
  void UpdateMsd(const AnalysisFrame& frame, rbmd::Id step);
  void UpdateVacf(const AnalysisFrame& frame, rbmd::Id step);
  void UpdateRdf(const AnalysisFrame& frame, rbmd::Id step);
  void FlushRdfFiles(bool force);
  void WriteRdfFile(const RdfPairState& pair_state) const;
  [[nodiscard]] double StepToTime(rbmd::Id delta_step) const;

  [[nodiscard]] bool ShouldSample(rbmd::Id interval, rbmd::Id step) const;
  [[nodiscard]] AnalysisSchedule BuildSchedule(rbmd::Id step) const;
  [[nodiscard]] size_t CurrentLocalAtomCount() const;

  bool _initialized{false};
  bool _mpi_root_only_mode{false};
  int _mpi_world_size{1};
  RdfConfig _rdf_config{};
  MsdConfig _msd_config{};
  VacfConfig _vacf_config{};
  bool _should_write_files{true};
  rbmd::Real _simulation_timestep{1};
  rbmd::Id _rdf_flush_interval{0};
  size_t _rdf_num_bins{0};
  size_t _rdf_worker_threads{1};
  std::vector<double> _rdf_bin_centers{};
  std::vector<double> _rdf_shell_volumes{};
  std::vector<RdfPairState> _rdf_pairs{};
  MsdDirectState _msd_state{};
  VacfDirectState _vacf_state{};

  std::vector<AnalysisHostFrame> _ring_buffer{};
  std::deque<AnalysisFrame> _pending_frames{};
  size_t _pending_frame_limit{4};
  size_t _ring_buffer_size{0};
  std::atomic<size_t> _write_index{0};
  std::atomic<size_t> _read_index{0};
  std::thread _output_thread{};
  std::mutex _mutex{};
  std::condition_variable _cv_not_full{};
  std::condition_variable _cv_not_empty{};
  std::atomic<bool> _stop_flag{false};
  STREAM _stream{nullptr};
  size_t _num_atoms{0};

  AnalysisFrame _mpi_local_frame{};
  AnalysisFrame _mpi_root_frame{};
  std::vector<int> _mpi_recv_counts{};
  std::vector<int> _mpi_recv_displs{};
  std::vector<rbmd::Real> _mpi_gathered_box_bounds{};
};
