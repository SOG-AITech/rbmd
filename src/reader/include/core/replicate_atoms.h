#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include "core/parsed_snapshot.h"

namespace reader::replication {
struct RemappedCoordinate {
  rbmd::Real value = 0.0;
  int image = 0;
};

inline std::size_t CheckedMultiply(std::size_t lhs, std::size_t rhs,
                            std::string_view name) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error(std::string("LAMMPS replicate overflow for ") +
                              std::string(name));
  }
  return lhs * rhs;
}

inline rbmd::Id CheckedIdAdd(rbmd::Id lhs, std::size_t rhs,
                      std::string_view name) {
  const auto max_id = static_cast<std::size_t>(
      std::numeric_limits<rbmd::Id>::max());
  if (rhs > max_id || static_cast<std::size_t>(lhs) > max_id - rhs) {
    throw std::overflow_error(std::string("LAMMPS replicate id overflow for ") +
                              std::string(name));
  }
  return static_cast<rbmd::Id>(static_cast<std::size_t>(lhs) + rhs);
}

inline rbmd::Id NoMoleculeId() {
  return static_cast<rbmd::Id>(-1);
}

inline int CheckedImageFlag(long long value) {
  if (value < static_cast<long long>(std::numeric_limits<int>::min()) ||
      value > static_cast<long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("LAMMPS replicate image flag overflow");
  }
  return static_cast<int>(value);
}

inline RemappedCoordinate RemapPeriodic(rbmd::Real value, rbmd::Real lo,
                                 rbmd::Real hi) {
  const rbmd::Real length = hi - lo;
  if (length <= rbmd::Real{0}) {
    return {value, 0};
  }

  long long image = static_cast<long long>(
      std::floor(static_cast<double>((value - lo) / length)));
  rbmd::Real wrapped = value - static_cast<rbmd::Real>(image) * length;
  while (wrapped < lo) {
    wrapped += length;
    --image;
  }
  while (wrapped >= hi) {
    wrapped -= length;
    ++image;
  }
  return {wrapped, CheckedImageFlag(image)};
}

inline rbmd::Id SameReplicaAtomId(rbmd::Id atom_id, std::size_t rep_index,
                           std::size_t atom_id_span) {
  return CheckedIdAdd(atom_id,
                      CheckedMultiply(rep_index, atom_id_span,
                                      "replicated atom id offset"),
                      "replicated atom id");
}

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

inline ParsedSnapshot::Atom BuildReplicatedAtom(const ParsedSnapshot::Box& seed_box,
                                         const ParsedSnapshot::Box& output_box,
                                         const ParsedSnapshot::Atom& atom,
                                         std::size_t rep_index,
                                         std::size_t atom_id_span,
                                         std::size_t molecule_offset,
                                         rbmd::Real dx,
                                         rbmd::Real dy,
                                         rbmd::Real dz,
                                         bool use_bond_periodic) {
  auto replica = atom;
  replica.id = SameReplicaAtomId(atom.id, rep_index, atom_id_span);
  if (replica.molecule != NoMoleculeId()) {
    replica.molecule =
        CheckedIdAdd(replica.molecule, molecule_offset, "molecule id");
  }

  const rbmd::Real xprd = seed_box.xhi - seed_box.xlo;
  const rbmd::Real yprd = seed_box.yhi - seed_box.ylo;
  const rbmd::Real zprd = seed_box.zhi - seed_box.zlo;
  if (use_bond_periodic) {
    replica.x = atom.x + dx;
    replica.y = atom.y + dy;
    replica.z = atom.z + dz;
  } else {
    replica.x = atom.x + static_cast<rbmd::Real>(atom.ix) * xprd + dx;
    replica.y = atom.y + static_cast<rbmd::Real>(atom.iy) * yprd + dy;
    replica.z = atom.z + static_cast<rbmd::Real>(atom.iz) * zprd + dz;
  }

  const auto x = RemapPeriodic(replica.x, output_box.xlo, output_box.xhi);
  const auto y = RemapPeriodic(replica.y, output_box.ylo, output_box.yhi);
  const auto z = RemapPeriodic(replica.z, output_box.zlo, output_box.zhi);
  replica.x = x.value;
  replica.y = y.value;
  replica.z = z.value;
  replica.ix = x.image;
  replica.iy = y.image;
  replica.iz = z.image;
  return replica;
}

// Iterate only replica tiles intersecting this rank's spatial subdomain.
// Candidate padding covers floating-point boundary rounding; ownership is
// checked by the same function used by the ordinary spatial reader.
template<class Owner, class Emit>
void ForEachLocalReplica(const ParsedSnapshot& seed,
                         const ParsedSnapshot::Box& output_box,
                         const std::array<std::size_t, 3>& factors,
                         const std::array<int, 3>& dims,
                         const std::array<int, 3>& coords,
                         std::size_t id_span, bool bond_periodic,
                         int rank, Owner owner, Emit emit) {
  const rbmd::Real lengths[3] = {seed.box.xhi - seed.box.xlo,
                                seed.box.yhi - seed.box.ylo,
                                seed.box.zhi - seed.box.zlo};
  const rbmd::Real low[3] = {seed.box.xlo, seed.box.ylo, seed.box.zlo};
  for (const auto& atom : seed.atoms) {
    const rbmd::Real pos[3] = {atom.x, atom.y, atom.z};
    const int images[3] = {atom.ix, atom.iy, atom.iz};
    long long begin[3], end[3], shift[3];
    for (int d = 0; d < 3; ++d) {
      const auto n = static_cast<long long>(factors[d]);
      if (!(lengths[d] > 0) || n <= 0 || dims[d] <= 0) {
        throw std::runtime_error("Invalid distributed replication dimensions");
      }
      const double fractional = (static_cast<double>(pos[d]) - low[d]) /
                                lengths[d] + (bond_periodic ? 0 : images[d]);
      if (!std::isfinite(fractional) || std::abs(fractional) > 1.e15) {
        throw std::runtime_error("Invalid seed coordinate for replication");
      }
      shift[d] = static_cast<long long>(std::floor(fractional));
      begin[d] = std::max(0LL, static_cast<long long>(std::floor(
          static_cast<double>(n) * coords[d] / dims[d])) - 2);
      end[d] = std::min(n, static_cast<long long>(std::ceil(
          static_cast<double>(n) * (coords[d] + 1) / dims[d])) + 2);
    }
    for (long long z = begin[2]; z < end[2]; ++z)
      for (long long y = begin[1]; y < end[1]; ++y)
        for (long long x = begin[0]; x < end[0]; ++x) {
          const long long tile[3] = {x, y, z};
          std::size_t original[3];
          for (int d = 0; d < 3; ++d) {
            const auto n = static_cast<long long>(factors[d]);
            auto wrapped = (tile[d] - shift[d]) % n;
            if (wrapped < 0) wrapped += n;
            original[d] = static_cast<std::size_t>(wrapped);
          }
          const auto replica_id = (original[2] * factors[1] + original[1]) *
                                  factors[0] + original[0];
          auto replica = BuildReplicatedAtom(seed.box, output_box, atom,
              replica_id, id_span, 0,
              static_cast<rbmd::Real>(original[0]) * lengths[0],
              static_cast<rbmd::Real>(original[1]) * lengths[1],
              static_cast<rbmd::Real>(original[2]) * lengths[2], bond_periodic);
          if (owner(replica) == rank) emit(replica);
        }
  }
}
}  // namespace reader::replication
