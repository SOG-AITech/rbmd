#pragma once

#include <array>

#include "../../common/types.h"

struct RBSOGLevelPreset {
  rbmd::Real b;
  rbmd::Real sigma_at_rc10;
  rbmd::Real target_accuracy;
  rbmd::Id mmax;
};

inline const std::array<RBSOGLevelPreset, 5>& GetRBSOGLevelPresets() {
  static const std::array<RBSOGLevelPreset, 5> kRBSOGLevelPresets = {{
      {static_cast<rbmd::Real>(2.0),
       static_cast<rbmd::Real>(5.027010924194599),
       static_cast<rbmd::Real>(2.289e-3),
       static_cast<rbmd::Id>(6)},
      {static_cast<rbmd::Real>(1.62976708826776469),
       static_cast<rbmd::Real>(3.633717409009413),
       static_cast<rbmd::Real>(1.158e-4),
       static_cast<rbmd::Id>(16)},
      {static_cast<rbmd::Real>(1.48783512395703226),
       static_cast<rbmd::Real>(2.662784519725113),
       static_cast<rbmd::Real>(1.142e-5),
       static_cast<rbmd::Id>(30)},
      {static_cast<rbmd::Real>(1.32070036405934420),
       static_cast<rbmd::Real>(2.277149356440992),
       static_cast<rbmd::Real>(5.583e-8),
       static_cast<rbmd::Id>(64)},
      {static_cast<rbmd::Real>(1.21812525709410644),
       static_cast<rbmd::Real>(1.774456369233284),
       static_cast<rbmd::Real>(3.389e-11),
       static_cast<rbmd::Id>(102)},
  }};
  return kRBSOGLevelPresets;
}
