#include "CompositeOutput.h"

#include <utility>

void CompositeOutput::Add(std::shared_ptr<Output> output) {
  if (output) {
    _outputs.push_back(std::move(output));
  }
}

void CompositeOutput::Init() {
  for (const auto& output : _outputs) {
    output->Init();
  }
}

void CompositeOutput::Execute(rbmd::Id current_timestep) {
  for (const auto& output : _outputs) {
    output->Execute(current_timestep);
  }
}
