#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace rbmd {

// The current molecular force-field dispatcher uses CVFF for OPLS dihedrals too.
// An explicit rule also supports OPLS systems without dihedral terms.
inline bool UseGeometricLJMixing(const std::string& dihedral_type,
                                 const std::string& mixing_rule) {
  if (mixing_rule.empty()) return dihedral_type == "OPLS";
  if (mixing_rule == "GEOMETRIC") return true;
  if (mixing_rule == "ARITHMETIC") return false;
  throw std::invalid_argument("force_field.mixing_rule must be geometric or arithmetic");
}

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RBMD_LJ_HD __host__ __device__
#else
#define RBMD_LJ_HD
#endif

template <typename Real>
RBMD_LJ_HD inline Real MixLJSigma(Real sigma_i, Real sigma_j, bool geometric) {
  return geometric ? std::sqrt(sigma_i * sigma_j)
                   : (sigma_i + sigma_j) * Real(0.5);
}

#undef RBMD_LJ_HD
}  // namespace rbmd
