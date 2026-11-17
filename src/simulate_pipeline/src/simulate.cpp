#include "common/revision_timing.h"
#include "common/revision_diagnostics.h"
#include "common/rbmd_define.h"
#include <cerrno>
#include <stdexcept>
#include "simulate.h"
#include "output/include/Logger.hpp"
#include "common/thermo_stats.hpp"
#include "common/timing_statistics.hpp"

#include <chrono>
#include <cstdlib>
#ifdef USE_MPI
#include <mpi.h>
#endif
rbmd::Id test_current_step = 0;
rbmd::Id test_num_steps = 0;

Simulate::Simulate(std::shared_ptr<Ensemble>& simulate_pipeline,
                   std::shared_ptr<Output>& output)
  : _simulate_pipeline(simulate_pipeline)
    , _output(output) {
  _time_step = DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
      "timestep", "execution"); //0.001
  _num_steps = DataManager::getInstance().getConfigData()->Get<rbmd::Id>(
      "num_steps", "execution"); //10000
  test_num_steps = _num_steps;
  _current_time = 0;
  _current_step = 0;
  ThermoStats::Instance().SetStep(_current_step);
}

void Simulate::Init() {
  rbmd::revision::Initialize();
  if (rbmd::revision::GetOptions().fixed_samples) _simulate_pipeline->InitForForceSampling();
  else _simulate_pipeline->Init();
  if (!rbmd::revision::GetOptions().fixed_samples) _output->Init();
  // Exclude force-field initialization and the first GPU warm-up execution
  // from production timing statistics.
  TimingStatistics::Instance().reset();
}

int Simulate::Execute() {
  using Clock = std::chrono::steady_clock;
  const auto& diagnostics = rbmd::revision::GetOptions();
  const bool frozen = diagnostics.fixed_samples > 0;
  const rbmd::Id evaluations = frozen ? diagnostics.fixed_samples : _num_steps;
  if (evaluations < 0) throw std::runtime_error("num_steps must be nonnegative");
  rbmd::Id warmup = 0;
  if (const char* env = std::getenv("RBMD_TIMING_WARMUP_STEPS")) {
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(env, &end, 10);
    if (errno || end == env || *end || parsed < 0 || parsed > evaluations)
      throw std::runtime_error("RBMD_TIMING_WARMUP_STEPS must be an integer in [0, number of evaluations]");
    warmup = static_cast<rbmd::Id>(parsed);
  }

  rbmd::revision::CaptureObservables(0);
  if (!frozen) {
    ThermoStats::Instance().OutputRow();
    _output->Execute(0);
  }
  rbmd::revision::DeviceSynchronize();
  auto window_start = Clock::now();
  while (_current_step < evaluations) {
    if (_current_step == warmup) {
      rbmd::revision::DeviceSynchronize();
      TimingStatistics::Instance().reset();
      window_start = Clock::now();
      Logger::Instance().info("RBMD_TIMING_WINDOW_START_STEP={}", _current_step);
    }
    ++_current_step;
    test_current_step = _current_step;
    ThermoStats::Instance().SetStep(_current_step);
    if (!frozen) _current_time += _time_step;

    // Device completion at both boundaries: no outstanding work from another step.
    if (diagnostics.isolated_timing) rbmd::revision::DeviceSynchronize();
    const auto step_start = Clock::now();
    if (frozen) _simulate_pipeline->EvaluateForces();
    else _simulate_pipeline->Run();
    if (diagnostics.isolated_timing) rbmd::revision::DeviceSynchronize();
    const double seconds = std::chrono::duration<double>(Clock::now() - step_start).count();
    TimingStatistics::Instance().record(frozen ? "Frozen-Evaluation-Wall" : (diagnostics.isolated_timing ? "Step-Wall" : "Step-HostSubmission-Wall"), seconds);

    rbmd::revision::CaptureObservables(_current_step);
    if (!frozen) {
      ThermoStats::Instance().OutputRow();
      _output->Execute(_current_step);
    }
  }
  rbmd::revision::DeviceSynchronize();
  if (warmup == evaluations) {
    TimingStatistics::Instance().reset();
    window_start = Clock::now();
  }
  const double local_wall = std::chrono::duration<double>(Clock::now() - window_start).count();
  double wall = local_wall;
#ifdef USE_MPI
  MPI_Allreduce(&local_wall, &wall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  Logger::Instance().info("RBMD_TIMING_WINDOW_END_STEP={}", _current_step);
  Logger::Instance().info("RBMD_TIMING_WINDOW_STEPS={}", evaluations - warmup);
  Logger::Instance().info("RBMD_TIMING_WINDOW_WALL_SECONDS={:.9f}", wall);
  rbmd::revision::WriteTiming();
  std::ostringstream context;
  context << "mode=" << (diagnostics.isolated_timing ? "isolated" : "production")
          << " fixed_configuration=" << frozen << " warmup=" << warmup
          << " measured_evaluations=" << evaluations - warmup
          << " loop_wall_max_rank_seconds=" << std::setprecision(17) << wall;
  TimingStatistics::Instance().print_summary(context.str());
  return 0;
}

bool Simulate::KeepGoing() { return _current_step < _num_steps; }
