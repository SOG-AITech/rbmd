#include "common/revision_timing.h"
#include "ensemble.h"
#include "common/revision_diagnostics.h"
#include "common/rbmd_define.h"
#include "common/timing_statistics.hpp"
#include <chrono>
extern rbmd::Id test_current_step;

void Ensemble::EvaluateForces() {
  const bool timing = rbmd::revision::GetOptions().isolated_timing;
  if (timing) rbmd::revision::DeviceSynchronize();
  const auto start = std::chrono::steady_clock::now();
  _force_controller->Execute();
  if (timing) {
    rbmd::revision::DeviceSynchronize();
    TimingStatistics::Instance().record("Force-Evaluation-Wall",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  }
  rbmd::revision::CaptureForces(test_current_step);
}

void Ensemble::InitForForceSampling() {
  // Load/wrap coordinates and initialize force state only. Thermostats,
  // constraints, barostats and velocity initialization must not alter the snapshot.
  _position_controller->Init();
  _position_controller->PBC();
  _force_controller->Init();
  EvaluateForces();
}
