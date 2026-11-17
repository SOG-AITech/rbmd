#pragma once
#include "revision_options.h"
#include <exception>
#include "rbmd_define.h"
#include "timing_statistics.hpp"

namespace rbmd { namespace revision {
inline void RequireGpu(ERROR_T status) {
  if (status != SUCCESS) throw std::runtime_error(std::string("diagnostic GPU operation failed: ") + GETERRORSTRING(status));
}
inline void DeviceSynchronize() { RequireGpu(DEVICESYNC()); }
// Opt-in, serialized GPU event interval. Only put device work between construction
// and Stop(): no MPI or host reductions. Transfers are allowed only in a
// separately named transfer category, never in a compute category.
class GpuTimer {
 public:
  explicit GpuTimer(const char* category, STREAM_T stream = nullptr)
      : category_(category), stream_(stream), enabled_(GetOptions().isolated_timing) {
    if (!enabled_) return;
    RequireGpu(DEVICESYNC());
    try {
      RequireGpu(NEW_EVENT(&begin_));
      RequireGpu(NEW_EVENT(&end_));
      RequireGpu(EVENT_RECORD(begin_, stream_));
    } catch (...) {
      if (begin_) EVENT_DESTROY(begin_);
      if (end_) EVENT_DESTROY(end_);
      throw;
    }
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;
  void Stop() {
    if (!enabled_ || stopped_) return;
    RequireGpu(EVENT_RECORD(end_, stream_));
    RequireGpu(EVENT_SYNC(end_));
    float ms = 0;
    RequireGpu(EVENT_ELAPSED_TIME(&ms, begin_, end_));
    TimingStatistics::Instance().record(category_, double(ms) * 1.e-3);
    stopped_ = true;
  }
  ~GpuTimer() {
    if (begin_) EVENT_DESTROY(begin_);
    if (end_) EVENT_DESTROY(end_);
  }
 private:
  const char* category_;
  STREAM_T stream_;
  bool enabled_, stopped_ = false;
  EVENT_T begin_ = nullptr, end_ = nullptr;
};
}} // namespace rbmd::revision

namespace rbmd { namespace revision {
class WorkflowTimer {
 public:
  explicit WorkflowTimer(const char* name) : name_(name), enabled_(GetOptions().isolated_timing) {
    if (enabled_) { RequireGpu(DEVICESYNC()); start_ = Clock::now(); }
  }
  ~WorkflowTimer() noexcept(false) {
    if (std::uncaught_exceptions()) return;
    if (!enabled_) return;
    RequireGpu(DEVICESYNC());
    TimingStatistics::Instance().record(name_, std::chrono::duration<double>(Clock::now() - start_).count());
  }
 private:
  using Clock = std::chrono::steady_clock;
  const char* name_;
  bool enabled_;
  Clock::time_point start_;
};
}}
