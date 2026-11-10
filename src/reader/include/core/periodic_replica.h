#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "core/parsed_snapshot.h"
namespace reader::replication {
// Follow one minimum-image bond into the appropriate replica tile.  Reusing
// the returned tile for the next bond preserves long dihedral chains too.
inline std::size_t BondTargetReplica(
    const ParsedSnapshot::Box& box, const ParsedSnapshot::Atom& anchor,
    const ParsedSnapshot::Atom& target, std::size_t anchor_replica,
    const std::array<std::size_t, 3>& factors) {
  const rbmd::Real lengths[] = {box.xhi-box.xlo, box.yhi-box.ylo,
                                box.zhi-box.zlo};
  const rbmd::Real from[] = {anchor.x, anchor.y, anchor.z};
  const rbmd::Real to[] = {target.x, target.y, target.z};
  std::size_t tile[3];
  for (int d=0; d<3; ++d) {
    if (!factors[d] || factors[d] > static_cast<std::size_t>(
            std::numeric_limits<int>::max()) || !(lengths[d]>0))
      throw std::runtime_error("Invalid periodic replication box/factors");
    const auto n=static_cast<long long>(factors[d]);
    const auto current=static_cast<long long>(anchor_replica % factors[d]);
    anchor_replica /= factors[d];
    const double delta=static_cast<double>(to[d])-from[d];
    const double ratio=delta/static_cast<double>(lengths[d]);
    if (!std::isfinite(ratio) || std::abs(ratio)>1.e15)
      throw std::runtime_error("Invalid periodic replication coordinate");
    // Preserve the sign at the exact half-box boundary (no image shift).
    const long long shift=ratio>0.5 ? -static_cast<long long>(std::ceil(ratio-0.5))
                           : ratio < -0.5 ? -static_cast<long long>(std::floor(ratio+0.5)) : 0;
    auto wrapped=(current+shift)%n;
    if (wrapped<0) wrapped+=n;
    tile[d]=static_cast<std::size_t>(wrapped);
  }
  return (tile[2]*factors[1]+tile[1])*factors[0]+tile[0];
}

}
