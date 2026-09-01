#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "types.h"

struct ThermoFrame {
  rbmd::Id step = 0;
  std::vector<std::string> ordered_keys;
  std::unordered_map<std::string, rbmd::Real> values;
  std::unordered_map<std::string, std::string> labels;
};

class IThermoView {
 public:
  virtual ~IThermoView() = default;

  virtual void OnFrame(const ThermoFrame& frame) = 0;
  virtual void OnReset() {}
  virtual void OnShutdown() {}
};
