#pragma once

#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace reader {

struct ParsedSnapshot {
  struct Box {
    rbmd::Real xlo = 0.0;
    rbmd::Real xhi = 0.0;
    rbmd::Real ylo = 0.0;
    rbmd::Real yhi = 0.0;
    rbmd::Real zlo = 0.0;
    rbmd::Real zhi = 0.0;
  };

  struct Topology {
    rbmd::Id atom_types = 0;
    rbmd::Id bond_types = 0;
    rbmd::Id angle_types = 0;
    rbmd::Id dihedral_types = 0;
    rbmd::Id improper_types = 0;
  };

  struct Atom {
    rbmd::Id id = 0;
    rbmd::Id molecule = 0;
    rbmd::Id type = 0;
    rbmd::Real charge = 0.0;
    rbmd::Real x = 0.0;
    rbmd::Real y = 0.0;
    rbmd::Real z = 0.0;
    rbmd::Real vx = 0.0;
    rbmd::Real vy = 0.0;
    rbmd::Real vz = 0.0;
    int ix = 0;
    int iy = 0;
    int iz = 0;
  };

  struct Bond {
    rbmd::Id id = 0;
    rbmd::Id type = 0;
    rbmd::Id atom1 = 0;
    rbmd::Id atom2 = 0;
  };

  struct Angle {
    rbmd::Id id = 0;
    rbmd::Id type = 0;
    rbmd::Id atom1 = 0;
    rbmd::Id atom2 = 0;
    rbmd::Id atom3 = 0;
  };

  struct Dihedral {
    rbmd::Id id = 0;
    rbmd::Id type = 0;
    rbmd::Id atom1 = 0;
    rbmd::Id atom2 = 0;
    rbmd::Id atom3 = 0;
    rbmd::Id atom4 = 0;
  };

  struct Improper {
    rbmd::Id id = 0;
    rbmd::Id type = 0;
    rbmd::Id atom1 = 0;
    rbmd::Id atom2 = 0;
    rbmd::Id atom3 = 0;
    rbmd::Id atom4 = 0;
  };

  struct PairCoeff {
    rbmd::Id type = 0;
    rbmd::Real epsilon = 0.0;
    rbmd::Real sigma = 0.0;
  };

  struct BondCoeff {
    rbmd::Id type = 0;
    rbmd::Real k = 0.0;
    rbmd::Real equilibrium = 0.0;
  };

  struct AngleCoeff {
    rbmd::Id type = 0;
    rbmd::Real k = 0.0;
    rbmd::Real equilibrium = 0.0;
  };

  struct DihedralCoeff {
    rbmd::Id type = 0;
    rbmd::Real k = 0.0;
    rbmd::Real sign = 0.0;
    rbmd::Real multiplicity = 0.0;
    rbmd::Real k1 = 0.0;
    rbmd::Real k2 = 0.0;
    rbmd::Real k3 = 0.0;
    rbmd::Real k4 = 0.0;
  };

  struct ImproperCoeff {
    rbmd::Id type = 0;
    rbmd::Real k = 0.0;
    rbmd::Real degree = 0.0;
    rbmd::Id d = 0.0;
    rbmd::Id n = 0.0;
  };

  Box box;
  Topology topology;
  std::unordered_map<rbmd::Id, rbmd::Real> masses;
  std::vector<Atom> atoms;
  std::vector<Bond> bonds;
  std::vector<Angle> angles;
  std::vector<Dihedral> dihedrals;
  std::vector<Improper> impropers;

  std::unordered_map<rbmd::Id, PairCoeff> pair_coeffs;
  std::unordered_map<rbmd::Id, BondCoeff> bond_coeffs;
  std::unordered_map<rbmd::Id, AngleCoeff> angle_coeffs;
  std::unordered_map<rbmd::Id, DihedralCoeff> dihedral_coeffs;
  std::unordered_map<rbmd::Id, ImproperCoeff> improper_coeffs;
};

}  // namespace reader
