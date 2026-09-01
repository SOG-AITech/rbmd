#pragma once

#include <memory>

class ConfigData;

class PostprocessRunner {
 public:
  void Run(const std::shared_ptr<ConfigData>& config) const;
};
