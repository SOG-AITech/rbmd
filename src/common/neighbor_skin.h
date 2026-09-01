#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "common/types.h"
#include "config_data.h"

namespace rbmd::neighbor {

inline rbmd::Real ReadSkinOrDefault(ConfigData* config) {
  if (config == nullptr ||
      !config->PathExists({"hyper_parameters", "neighbor", "skin"})) {
    return rbmd::Real(0);
  }

  const auto skin =
      config->Get<rbmd::Real>("skin", "hyper_parameters", "neighbor");
  if (!std::isfinite(static_cast<double>(skin)) || skin < rbmd::Real(0)) {
    std::ostringstream oss;
    oss << "Neighbor skin must be finite and non-negative. Current value: "
        << skin;
    throw std::runtime_error(oss.str());
  }

  return skin;
}

inline rbmd::Real NeighborCutoff(rbmd::Real cutoff, rbmd::Real skin,
                                 const char* context) {
  if (!std::isfinite(static_cast<double>(cutoff)) || cutoff <= rbmd::Real(0)) {
    std::ostringstream oss;
    oss << context << " invalid cut_off=" << cutoff;
    throw std::runtime_error(oss.str());
  }

  const auto neighbor_cutoff = cutoff + skin;
  if (!std::isfinite(static_cast<double>(neighbor_cutoff)) ||
      neighbor_cutoff <= rbmd::Real(0)) {
    std::ostringstream oss;
    oss << context << " invalid neighbor cutoff=" << neighbor_cutoff
        << ", cut_off=" << cutoff << ", skin=" << skin;
    throw std::runtime_error(oss.str());
  }

  return neighbor_cutoff;
}

inline void ValidateHaloCutoffCoversNeighborCutoff(rbmd::Real halo_cutoff,
                                                   rbmd::Real neighbor_cutoff,
                                                   const char* context) {
  if (!std::isfinite(static_cast<double>(halo_cutoff)) ||
      halo_cutoff <= rbmd::Real(0)) {
    std::ostringstream oss;
    oss << context << " invalid halo cutoff=" << halo_cutoff;
    throw std::runtime_error(oss.str());
  }
  if (!std::isfinite(static_cast<double>(neighbor_cutoff)) ||
      neighbor_cutoff <= rbmd::Real(0)) {
    std::ostringstream oss;
    oss << context << " invalid neighbor cutoff=" << neighbor_cutoff;
    throw std::runtime_error(oss.str());
  }
  if (halo_cutoff < neighbor_cutoff) {
    std::ostringstream oss;
    oss << context << " halo cutoff must cover neighbor cutoff: halo_cutoff="
        << halo_cutoff << ", neighbor_cutoff=" << neighbor_cutoff;
    throw std::runtime_error(oss.str());
  }
}

}  // namespace rbmd::neighbor
