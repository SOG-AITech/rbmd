#pragma once

#include <cstdint>

// Fixed field bundles forwarded through the persistent three-stage halo plan.
// The names describe the consumer stage; the communication layer owns only
// packing/unpacking and periodic coordinate shifts.
enum class ForwardGhostState : std::uint8_t {
  Coordinates = 0,
  PredictedPositionForward,
  CorrectedStateForward,
  VelocityForward,
  VelocityCorrectedForward,
};
