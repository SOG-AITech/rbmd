#pragma once

#include <memory>
#include <vector>

#include "output.h"

class CompositeOutput : public Output {
 public:
  CompositeOutput() = default;
  ~CompositeOutput() override = default;

  void Add(std::shared_ptr<Output> output);

  void Init() override;
  void Execute(rbmd::Id current_timestep) override;

 private:
  std::vector<std::shared_ptr<Output>> _outputs;
};
