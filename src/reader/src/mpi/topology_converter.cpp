#include "mpi/topology_converter.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

#include "formats/structure_adapters.h"

namespace reader::mpi {

namespace {

template <typename T>
void FreeArray(T*& ptr) {
  if (ptr) {
    std::free(ptr);
    ptr = nullptr;
  }
}

void ResetPerAtomTopology(FullStructureData& output) {
  FreeArray(output._h_num_bond);
  FreeArray(output._h_bond_type_per_atom);
  FreeArray(output._h_bond_atom_per_atom);

  FreeArray(output._h_num_angle);
  FreeArray(output._h_angle_type_per_atom);
  FreeArray(output._h_angle_atom1_per_atom);
  FreeArray(output._h_angle_atom2_per_atom);
  FreeArray(output._h_angle_atom3_per_atom);

  FreeArray(output._h_num_dihedral);
  FreeArray(output._h_dihedral_type_per_atom);
  FreeArray(output._h_dihedral_atom1_per_atom);
  FreeArray(output._h_dihedral_atom2_per_atom);
  FreeArray(output._h_dihedral_atom3_per_atom);
  FreeArray(output._h_dihedral_atom4_per_atom);

  FreeArray(output._h_num_improper);
  FreeArray(output._h_improper_type_per_atom);
  FreeArray(output._h_improper_atom1_per_atom);
  FreeArray(output._h_improper_atom2_per_atom);
  FreeArray(output._h_improper_atom3_per_atom);
  FreeArray(output._h_improper_atom4_per_atom);
}

}  // namespace

void ConvertGlobalToPerAtomTopology(const ParsedSnapshot& snapshot,
                                    FullStructureData& output,
                                    const DeviceData& device_data) {
  ResetPerAtomTopology(output);

  const rbmd::Id local_atom_count =
      static_cast<rbmd::Id>(snapshot.atoms.size());
  const rbmd::Id nmax = std::max(device_data.nmax, local_atom_count);
  if (nmax <= 0) {
    return;
  }

  const int bond_per_atom = std::max(device_data.bond_per_atom, 0);
  const int angle_per_atom = std::max(device_data.angle_per_atom, 0);
  const int dihedral_per_atom = std::max(device_data.dihedral_per_atom, 0);
  const int improper_per_atom = std::max(device_data.improper_per_atom, 0);

  const std::size_t nmax_size = static_cast<std::size_t>(nmax);

  output._h_num_bond = detail::AllocateStructureArray<int>(nmax_size);
  output._h_bond_type_per_atom =
      detail::AllocateStructureArray<int>(nmax_size *
          static_cast<std::size_t>(bond_per_atom));
  output._h_bond_atom_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(bond_per_atom));

  output._h_num_angle = detail::AllocateStructureArray<int>(nmax_size);
  output._h_angle_type_per_atom =
      detail::AllocateStructureArray<int>(nmax_size *
          static_cast<std::size_t>(angle_per_atom));
  output._h_angle_atom1_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(angle_per_atom));
  output._h_angle_atom2_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(angle_per_atom));
  output._h_angle_atom3_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(angle_per_atom));

  output._h_num_dihedral = detail::AllocateStructureArray<int>(nmax_size);
  output._h_dihedral_type_per_atom =
      detail::AllocateStructureArray<int>(nmax_size *
          static_cast<std::size_t>(dihedral_per_atom));
  output._h_dihedral_atom1_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(dihedral_per_atom));
  output._h_dihedral_atom2_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(dihedral_per_atom));
  output._h_dihedral_atom3_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(dihedral_per_atom));
  output._h_dihedral_atom4_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(dihedral_per_atom));

  output._h_num_improper = detail::AllocateStructureArray<int>(nmax_size);
  output._h_improper_type_per_atom =
      detail::AllocateStructureArray<int>(nmax_size *
          static_cast<std::size_t>(improper_per_atom));
  output._h_improper_atom1_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(improper_per_atom));
  output._h_improper_atom2_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(improper_per_atom));
  output._h_improper_atom3_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(improper_per_atom));
  output._h_improper_atom4_per_atom =
      detail::AllocateStructureArray<rbmd::Id>(nmax_size *
          static_cast<std::size_t>(improper_per_atom));

  detail::FillZero(output._h_num_bond, nmax_size);
  detail::FillZero(output._h_bond_type_per_atom,
                   nmax_size * static_cast<std::size_t>(bond_per_atom));
  detail::FillZero(output._h_bond_atom_per_atom,
                   nmax_size * static_cast<std::size_t>(bond_per_atom));

  detail::FillZero(output._h_num_angle, nmax_size);
  detail::FillZero(output._h_angle_type_per_atom,
                   nmax_size * static_cast<std::size_t>(angle_per_atom));
  detail::FillZero(output._h_angle_atom1_per_atom,
                   nmax_size * static_cast<std::size_t>(angle_per_atom));
  detail::FillZero(output._h_angle_atom2_per_atom,
                   nmax_size * static_cast<std::size_t>(angle_per_atom));
  detail::FillZero(output._h_angle_atom3_per_atom,
                   nmax_size * static_cast<std::size_t>(angle_per_atom));

  detail::FillZero(output._h_num_dihedral, nmax_size);
  detail::FillZero(output._h_dihedral_type_per_atom,
                   nmax_size * static_cast<std::size_t>(dihedral_per_atom));
  detail::FillZero(output._h_dihedral_atom1_per_atom,
                   nmax_size * static_cast<std::size_t>(dihedral_per_atom));
  detail::FillZero(output._h_dihedral_atom2_per_atom,
                   nmax_size * static_cast<std::size_t>(dihedral_per_atom));
  detail::FillZero(output._h_dihedral_atom3_per_atom,
                   nmax_size * static_cast<std::size_t>(dihedral_per_atom));
  detail::FillZero(output._h_dihedral_atom4_per_atom,
                   nmax_size * static_cast<std::size_t>(dihedral_per_atom));

  detail::FillZero(output._h_num_improper, nmax_size);
  detail::FillZero(output._h_improper_type_per_atom,
                   nmax_size * static_cast<std::size_t>(improper_per_atom));
  detail::FillZero(output._h_improper_atom1_per_atom,
                   nmax_size * static_cast<std::size_t>(improper_per_atom));
  detail::FillZero(output._h_improper_atom2_per_atom,
                   nmax_size * static_cast<std::size_t>(improper_per_atom));
  detail::FillZero(output._h_improper_atom3_per_atom,
                   nmax_size * static_cast<std::size_t>(improper_per_atom));
  detail::FillZero(output._h_improper_atom4_per_atom,
                   nmax_size * static_cast<std::size_t>(improper_per_atom));

  std::unordered_map<rbmd::Id, int> gid_to_local;
  gid_to_local.reserve(snapshot.atoms.size());
  for (std::size_t i = 0; i < snapshot.atoms.size(); ++i) {
    gid_to_local[snapshot.atoms[i].id] = static_cast<int>(i);
  }

  auto append_bond = [&](rbmd::Id atom_gid,
                         rbmd::Id partner_gid,
                         rbmd::Id type_gid) {
    auto it = gid_to_local.find(atom_gid);
    if (it == gid_to_local.end()) {
      return;
    }
    if (bond_per_atom <= 0) {
      throw std::runtime_error("bond_per_atom must be > 0");
    }
    const int index = it->second;
    int count = output._h_num_bond[index];
    if (count >= bond_per_atom) {
      throw std::runtime_error("bond_per_atom overflow");
    }
    const std::size_t offset = static_cast<std::size_t>(index) *
        static_cast<std::size_t>(bond_per_atom) +
        static_cast<std::size_t>(count);
    output._h_bond_type_per_atom[offset] = static_cast<int>(type_gid);
    output._h_bond_atom_per_atom[offset] = partner_gid;
    output._h_num_bond[index] = count + 1;
  };

  auto append_angle = [&](rbmd::Id atom_gid,
                          const ParsedSnapshot::Angle& angle) {
    auto it = gid_to_local.find(atom_gid);
    if (it == gid_to_local.end()) {
      return;
    }
    if (angle_per_atom <= 0) {
      throw std::runtime_error("angle_per_atom must be > 0");
    }
    const int index = it->second;
    int count = output._h_num_angle[index];
    if (count >= angle_per_atom) {
      throw std::runtime_error("angle_per_atom overflow");
    }
    const std::size_t offset = static_cast<std::size_t>(index) *
        static_cast<std::size_t>(angle_per_atom) +
        static_cast<std::size_t>(count);
    output._h_angle_type_per_atom[offset] = static_cast<int>(angle.type);
    output._h_angle_atom1_per_atom[offset] = angle.atom1;
    output._h_angle_atom2_per_atom[offset] = angle.atom2;
    output._h_angle_atom3_per_atom[offset] = angle.atom3;
    output._h_num_angle[index] = count + 1;
  };

  auto append_dihedral = [&](rbmd::Id atom_gid,
                             const ParsedSnapshot::Dihedral& dihedral) {
    auto it = gid_to_local.find(atom_gid);
    if (it == gid_to_local.end()) {
      return;
    }
    if (dihedral_per_atom <= 0) {
      throw std::runtime_error("dihedral_per_atom must be > 0");
    }
    const int index = it->second;
    int count = output._h_num_dihedral[index];
    if (count >= dihedral_per_atom) {
      throw std::runtime_error("dihedral_per_atom overflow");
    }
    const std::size_t offset = static_cast<std::size_t>(index) *
        static_cast<std::size_t>(dihedral_per_atom) +
        static_cast<std::size_t>(count);
    output._h_dihedral_type_per_atom[offset] = static_cast<int>(dihedral.type);
    output._h_dihedral_atom1_per_atom[offset] = dihedral.atom1;
    output._h_dihedral_atom2_per_atom[offset] = dihedral.atom2;
    output._h_dihedral_atom3_per_atom[offset] = dihedral.atom3;
    output._h_dihedral_atom4_per_atom[offset] = dihedral.atom4;
    output._h_num_dihedral[index] = count + 1;
  };

  auto append_improper = [&](rbmd::Id atom_gid,
                             const ParsedSnapshot::Improper& improper) {
    auto it = gid_to_local.find(atom_gid);
    if (it == gid_to_local.end()) {
      return;
    }
    if (improper_per_atom <= 0) {
      throw std::runtime_error("improper_per_atom must be > 0");
    }
    const int index = it->second;
    int count = output._h_num_improper[index];
    if (count >= improper_per_atom) {
      throw std::runtime_error("improper_per_atom overflow");
    }
    const std::size_t offset = static_cast<std::size_t>(index) *
        static_cast<std::size_t>(improper_per_atom) +
        static_cast<std::size_t>(count);
    output._h_improper_type_per_atom[offset] = static_cast<int>(improper.type);
    output._h_improper_atom1_per_atom[offset] = improper.atom1;
    output._h_improper_atom2_per_atom[offset] = improper.atom2;
    output._h_improper_atom3_per_atom[offset] = improper.atom3;
    output._h_improper_atom4_per_atom[offset] = improper.atom4;
    output._h_num_improper[index] = count + 1;
  };

  for (const auto& bond : snapshot.bonds) {
    append_bond(bond.atom1, bond.atom2, bond.type);
    append_bond(bond.atom2, bond.atom1, bond.type);
  }

  for (const auto& angle : snapshot.angles) {
    append_angle(angle.atom1, angle);
    append_angle(angle.atom2, angle);
    append_angle(angle.atom3, angle);
  }

  for (const auto& dihedral : snapshot.dihedrals) {
    append_dihedral(dihedral.atom1, dihedral);
    append_dihedral(dihedral.atom2, dihedral);
    append_dihedral(dihedral.atom3, dihedral);
    append_dihedral(dihedral.atom4, dihedral);
  }

  for (const auto& improper : snapshot.impropers) {
    append_improper(improper.atom1, improper);
    append_improper(improper.atom2, improper);
    append_improper(improper.atom3, improper);
    append_improper(improper.atom4, improper);
  }
}

}  // namespace reader::mpi
