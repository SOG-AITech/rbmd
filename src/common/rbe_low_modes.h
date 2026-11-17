#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace rbmd {
// Host-only, independent draws from the discrete Gaussian outside a physical
// wave-number sphere. Cache O(nx*ny) slices, not an O(nx*ny*nz) mode table.
// The numerical Gaussian support uses k_d^2/(4*alpha) <= 48 (rounded up).
class RbeLowModes {
 public:
  struct Mode { int x, y, z; double weight; };
  bool Setup(const std::array<double, 3>& lengths, double alpha, double cutoff) {
    if (ready_ && lengths == lengths_ && alpha == alpha_ && cutoff == cutoff_)
      return false;
    if (!(alpha > 0) || !std::isfinite(alpha) || cutoff < 0 || !std::isfinite(cutoff))
      throw std::invalid_argument("RBE low modes require finite alpha > 0 and K_max >= 0");
    ready_ = false;
    low_.clear(); slices_.clear();
    std::array<double, 3> unit;
    std::array<int, 3> bound, low_bound;
    double slice_count = 1, low_candidates = 1;
    for (int d = 0; d < 3; ++d) {
      if (!(lengths[d] > 0) || !std::isfinite(lengths[d]))
        throw std::invalid_argument("RBE low modes require finite positive box lengths");
      unit[d] = 2 * std::acos(-1.0) / lengths[d];
      const double b = std::ceil(std::sqrt(192 * alpha) / unit[d]);
      const double lb = std::floor(cutoff / unit[d]);
      if (!std::isfinite(b) || b > 100000 || lb > 100000)
        throw std::invalid_argument("RBE low-mode grid is too large");
      bound[d] = static_cast<int>(b);
      low_bound[d] = static_cast<int>(lb);
      if (d < 2) slice_count *= 2 * b + 1;
      low_candidates *= 2 * lb + 1;
    }
    if (slice_count > 4000000 || low_candidates > 10000000)
      throw std::invalid_argument("RBE low-mode grid exceeds host memory/work limit");
    const double cut2 = cutoff * cutoff;
    auto k2 = [&](int x, int y, int z) {
      return std::pow(x * unit[0], 2) + std::pow(y * unit[1], 2) + std::pow(z * unit[2], 2);
    };
    for (int x = -low_bound[0]; x <= low_bound[0]; ++x)
      for (int y = -low_bound[1]; y <= low_bound[1]; ++y)
        for (int z = -low_bound[2]; z <= low_bound[2]; ++z)
          if ((x || y || z) && k2(x, y, z) <= cut2)
            low_.push_back({x, y, z, std::exp(-k2(x, y, z) / (4 * alpha))});

    // Suffix mass for |nz|, including both signs. Avoid S_total - S_low
    // cancellation when nearly all Gaussian mass is deterministic.
    z_mass_.assign(bound[2] + 2, 0);
    for (int z = bound[2]; z >= 0; --z)
      z_mass_[z] = z_mass_[z + 1] + (z ? 2 : 1) *
          std::exp(-std::pow(z * unit[2], 2) / (4 * alpha));
    std::vector<double> masses;
    tail_mass_ = 0;
    for (int x = -bound[0]; x <= bound[0]; ++x) {
      for (int y = -bound[1]; y <= bound[1]; ++y) {
        const double xy2 = k2(x, y, 0);
        int zmin = xy2 > cut2 ? 0 :
            static_cast<int>(std::floor(std::sqrt(std::max(0.0, cut2 - xy2)) / unit[2])) + 1;
        // Resolve rounding at the inclusive sphere boundary using the same predicate.
        zmin = std::min(zmin, bound[2] + 1);
        while (zmin > 0 && k2(x, y, zmin - 1) > cut2) --zmin;
        while (zmin <= bound[2] && k2(x, y, zmin) <= cut2) ++zmin;
        if (zmin > bound[2]) continue;
        const double mass = std::exp(-xy2 / (4 * alpha)) * z_mass_[zmin];
        if (mass == 0) continue;
        slices_.push_back({x, y, zmin});
        masses.push_back(mass);
        tail_mass_ += mass;
      }
    }
    if (!(tail_mass_ > 0) || !std::isfinite(tail_mass_))
      throw std::invalid_argument("RBE K_max leaves no numerically representable Gaussian tail; reduce K_max");
    slice_distribution_ = std::discrete_distribution<std::size_t>(masses.begin(), masses.end());
    lengths_ = lengths; alpha_ = alpha; cutoff_ = cutoff; ready_ = true;
    return true;
  }

  template<class Generator>
  Mode DrawTail(Generator& generator, int batch_size) {
    if (!ready_ || batch_size <= 0) throw std::invalid_argument("Invalid RBE tail sampler state/batch size");
    const auto& slice = slices_[slice_distribution_(generator)];
    const double u = std::generate_canonical<double, 53>(generator) * z_mass_[slice.zmin];
    int lo = slice.zmin, hi = static_cast<int>(z_mass_.size()) - 2;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (z_mass_[mid + 1] > u) lo = mid + 1;
      else hi = mid;
    }
    const int z = lo && std::uniform_int_distribution<int>(0, 1)(generator) ? -lo : lo;
    return {slice.x, slice.y, z, tail_mass_ / batch_size};
  }
  const std::vector<Mode>& Low() const { return low_; }
  double TailMass() const { return tail_mass_; }

 private:
  struct Slice { int x, y, zmin; };
  bool ready_ = false;
  std::array<double, 3> lengths_{};
  double alpha_ = 0, cutoff_ = 0, tail_mass_ = 0;
  std::vector<Mode> low_;
  std::vector<Slice> slices_;
  std::vector<double> z_mass_;
  std::discrete_distribution<std::size_t> slice_distribution_;
};
}  // namespace rbmd
