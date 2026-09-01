#include "../include/model/device_data.h"
#include <cstddef>

void DeviceData::ResizeTopology(rbmd::Id nmax) {
  const rbmd::Id safe_nmax = nmax > 0 ? nmax : 0;
  const int safe_bond_per_atom = bond_per_atom > 0 ? bond_per_atom : 0;
  const int safe_angle_per_atom = angle_per_atom > 0 ? angle_per_atom : 0;
  const int safe_dihedral_per_atom = dihedral_per_atom > 0 ? dihedral_per_atom : 0;
  const int safe_improper_per_atom = improper_per_atom > 0 ? improper_per_atom : 0;
  const int safe_maxspecial = maxspecial > 0 ? maxspecial : 0;

  this->nmax = safe_nmax;

  const std::size_t nmax_size = static_cast<std::size_t>(safe_nmax);

  d_num_bond.resize(nmax_size);
  d_bond_type.resize(nmax_size * static_cast<std::size_t>(safe_bond_per_atom));
  d_bond_atom.resize(nmax_size * static_cast<std::size_t>(safe_bond_per_atom));

  d_num_angle.resize(nmax_size);
  d_angle_type.resize(nmax_size * static_cast<std::size_t>(safe_angle_per_atom));
  d_angle_atom1.resize(nmax_size * static_cast<std::size_t>(safe_angle_per_atom));
  d_angle_atom2.resize(nmax_size * static_cast<std::size_t>(safe_angle_per_atom));
  d_angle_atom3.resize(nmax_size * static_cast<std::size_t>(safe_angle_per_atom));

  d_num_dihedral.resize(nmax_size);
  d_dihedral_type.resize(nmax_size * static_cast<std::size_t>(safe_dihedral_per_atom));
  d_dihedral_atom1.resize(nmax_size * static_cast<std::size_t>(safe_dihedral_per_atom));
  d_dihedral_atom2.resize(nmax_size * static_cast<std::size_t>(safe_dihedral_per_atom));
  d_dihedral_atom3.resize(nmax_size * static_cast<std::size_t>(safe_dihedral_per_atom));
  d_dihedral_atom4.resize(nmax_size * static_cast<std::size_t>(safe_dihedral_per_atom));

  d_num_improper.resize(nmax_size);
  d_improper_type.resize(nmax_size * static_cast<std::size_t>(safe_improper_per_atom));
  d_improper_atom1.resize(nmax_size * static_cast<std::size_t>(safe_improper_per_atom));
  d_improper_atom2.resize(nmax_size * static_cast<std::size_t>(safe_improper_per_atom));
  d_improper_atom3.resize(nmax_size * static_cast<std::size_t>(safe_improper_per_atom));
  d_improper_atom4.resize(nmax_size * static_cast<std::size_t>(safe_improper_per_atom));

  d_nspecial.resize(nmax_size * 3);
  d_special.resize(nmax_size * static_cast<std::size_t>(safe_maxspecial));
}
