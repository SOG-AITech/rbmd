#pragma once
#include "revision_options.h"
#include "types.h"

namespace rbmd { namespace revision {
void Initialize();
void CaptureForces(rbmd::Id step);
void CaptureObservables(rbmd::Id step);
void WriteTiming();
double LocalKineticEnergy();
void AddThermostatWork(double work);
// Bracket only the actual velocity scaling (including COM bias removal/restore).
// This measures discrete work delivered to atoms, positive for heating.
class ThermostatWork {
 public:
  ThermostatWork() : enabled_(GetOptions().thermostat_heat), before_(enabled_ ? LocalKineticEnergy() : 0) {}
  ~ThermostatWork() = default;
  void Finish() {
    if (enabled_) { AddThermostatWork(LocalKineticEnergy() - before_); enabled_ = false; }
  }
 private:
  bool enabled_;
  double before_;
};
}}
