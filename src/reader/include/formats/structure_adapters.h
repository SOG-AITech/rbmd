#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/parsed_snapshot.h"
#include "structure_data/atoms_structure_data.h"
#include "structure_data/basic_structure_data.h"
#include "structure_data/charge_structure_data.h"
#include "structure_data/full_structure_data.h"
#include "structure_data/structure_data.h"

namespace reader {

namespace detail {

template <typename T>
inline T* AllocateStructureArray(std::size_t count) {
  if (count == 0) {
    return nullptr;
  }
  auto* ptr = static_cast<T*>(std::malloc(sizeof(T) * count));   //TODO MALLOC HOST
  if (!ptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

template <typename T>
inline void FillZero(T* ptr, std::size_t count) {
  if (!ptr || count == 0) {
    return;
  }
  std::fill_n(ptr, count, T{});
}

inline void PopulateCommonStructure(StructureData& structure,
                                    const ParsedSnapshot& snapshot) {
  const auto atom_count = snapshot.atoms.size();

  structure._h_px = AllocateStructureArray<rbmd::Real>(atom_count);
  structure._h_py = AllocateStructureArray<rbmd::Real>(atom_count);
  structure._h_pz = AllocateStructureArray<rbmd::Real>(atom_count);
  structure._h_flagX = AllocateStructureArray<rbmd::Id>(atom_count);
  structure._h_flagY = AllocateStructureArray<rbmd::Id>(atom_count);
  structure._h_flagZ = AllocateStructureArray<rbmd::Id>(atom_count);
  structure._h_vx = AllocateStructureArray<rbmd::Real>(atom_count);
  structure._h_vy = AllocateStructureArray<rbmd::Real>(atom_count);
  structure._h_vz = AllocateStructureArray<rbmd::Real>(atom_count);
  structure._h_atoms_id = AllocateStructureArray<rbmd::Id>(atom_count);
  structure._h_atoms_type = AllocateStructureArray<rbmd::Id>(atom_count);
  structure._h_molecular_id = AllocateStructureArray<rbmd::Id>(atom_count);

  FillZero(structure._h_vx, atom_count);
  FillZero(structure._h_vy, atom_count);
  FillZero(structure._h_vz, atom_count);
  FillZero(structure._h_molecular_id, atom_count);

  for (std::size_t i = 0; i < atom_count; ++i) {
    const auto& atom = snapshot.atoms[i];
    structure._h_px[i] = atom.x;
    structure._h_py[i] = atom.y;
    structure._h_pz[i] = atom.z;
    structure._h_vx[i] = atom.vx;
    structure._h_vy[i] = atom.vy;
    structure._h_vz[i] = atom.vz;
    structure._h_atoms_id[i] = atom.id;
    structure._h_atoms_type[i] = atom.type;
    structure._h_flagX[i] = static_cast<rbmd::Id>(atom.ix);
    structure._h_flagY[i] = static_cast<rbmd::Id>(atom.iy);
    structure._h_flagZ[i] = static_cast<rbmd::Id>(atom.iz);
    if (structure._h_molecular_id) {
      structure._h_molecular_id[i] = atom.molecule;
    }
  }
}

}  // namespace detail

class StructureAdapter {
 public:
  virtual ~StructureAdapter() = default;
  virtual void Populate(const ParsedSnapshot& snapshot) = 0;
};

class AtomsStructureAdapter : public StructureAdapter {
 public:
  explicit AtomsStructureAdapter(std::shared_ptr<StructureData> data)
      : data_(std::move(data)) {}

  void Populate(const ParsedSnapshot& snapshot) override {
    if (!data_) {
      throw std::runtime_error("AtomsStructureAdapter requires valid data pointer");
    }
    detail::PopulateCommonStructure(*data_, snapshot);
  }

 protected:
  const std::shared_ptr<StructureData>& data() const { return data_; }

 private:
  std::shared_ptr<StructureData> data_;
};

class BasicStructureAdapter : public AtomsStructureAdapter {
 public:
  explicit BasicStructureAdapter(std::shared_ptr<BasicStructureData> data)
      : AtomsStructureAdapter(std::static_pointer_cast<StructureData>(data)),
        basic_data_(std::move(data)) {}

  void Populate(const ParsedSnapshot& snapshot) override {
    AtomsStructureAdapter::Populate(snapshot);
    if (!basic_data_) {
      throw std::runtime_error("BasicStructureAdapter requires valid data pointer");
    }
    basic_data_->_h_charge = nullptr;
  }

 protected:
  const std::shared_ptr<BasicStructureData>& basic() const { return basic_data_; }

 private:
  std::shared_ptr<BasicStructureData> basic_data_;
};

class ChargeStructureAdapter : public BasicStructureAdapter {
 public:
  explicit ChargeStructureAdapter(std::shared_ptr<ChargeStructureData> data)
      : BasicStructureAdapter(std::static_pointer_cast<BasicStructureData>(data)),
        charge_data_(std::move(data)) {}

  void Populate(const ParsedSnapshot& snapshot) override {
    BasicStructureAdapter::Populate(snapshot);
    if (!charge_data_) {
      throw std::runtime_error("ChargeStructureAdapter requires valid data pointer");
    }
    const auto atom_count = snapshot.atoms.size();
    charge_data_->_h_charge = detail::AllocateStructureArray<rbmd::Real>(atom_count);
    for (std::size_t i = 0; i < atom_count; ++i) {
      charge_data_->_h_charge[i] = snapshot.atoms[i].charge;
    }
  }

 private:
  std::shared_ptr<ChargeStructureData> charge_data_;
};

class FullStructureAdapter : public BasicStructureAdapter {
 public:
  explicit FullStructureAdapter(std::shared_ptr<FullStructureData> data)
      : BasicStructureAdapter(std::static_pointer_cast<BasicStructureData>(data)),
        full_data_(std::move(data)) {}

  void Populate(const ParsedSnapshot& snapshot) override {
    BasicStructureAdapter::Populate(snapshot);
    if (!full_data_) {
      throw std::runtime_error("FullStructureAdapter requires valid data pointer");
    }

    const auto atom_count = snapshot.atoms.size();
    full_data_->_h_charge = detail::AllocateStructureArray<rbmd::Real>(atom_count);
    full_data_->_h_molecules_id = detail::AllocateStructureArray<rbmd::Id>(atom_count);
    for (std::size_t i = 0; i < atom_count; ++i) {
      full_data_->_h_charge[i] = snapshot.atoms[i].charge;
      full_data_->_h_molecules_id[i] = snapshot.atoms[i].molecule;
    }

    const auto bond_count = snapshot.bonds.size();
    full_data_->_h_bond_type = detail::AllocateStructureArray<rbmd::Id>(bond_count);
    full_data_->_h_bond_id0 = detail::AllocateStructureArray<rbmd::Id>(bond_count);
    full_data_->_h_bond_id1 = detail::AllocateStructureArray<rbmd::Id>(bond_count);
    for (std::size_t i = 0; i < bond_count; ++i) {
      const auto& bond = snapshot.bonds[i];
      full_data_->_h_bond_type[i] = bond.type;
      full_data_->_h_bond_id0[i] = bond.atom1;
      full_data_->_h_bond_id1[i] = bond.atom2;
    }

    const auto angle_count = snapshot.angles.size();
    full_data_->_h_angle_type = detail::AllocateStructureArray<rbmd::Id>(angle_count);
    full_data_->_h_angle_id0 = detail::AllocateStructureArray<rbmd::Id>(angle_count);
    full_data_->_h_angle_id1 = detail::AllocateStructureArray<rbmd::Id>(angle_count);
    full_data_->_h_angle_id2 = detail::AllocateStructureArray<rbmd::Id>(angle_count);
    full_data_->_h_angle_id_vec = detail::AllocateStructureArray<Id3>(angle_count);
    for (std::size_t i = 0; i < angle_count; ++i) {
      const auto& angle = snapshot.angles[i];
      full_data_->_h_angle_type[i] = angle.type;
      full_data_->_h_angle_id0[i] = angle.atom1;
      full_data_->_h_angle_id1[i] = angle.atom2;
      full_data_->_h_angle_id2[i] = angle.atom3;
      full_data_->_h_angle_id_vec[i].x = angle.atom1;
      full_data_->_h_angle_id_vec[i].y = angle.atom2;
      full_data_->_h_angle_id_vec[i].z = angle.atom3;
    }

    const auto dihedral_count = snapshot.dihedrals.size();
    full_data_->_h_dihedral_type =
        detail::AllocateStructureArray<rbmd::Id>(dihedral_count);
    full_data_->_h_dihedral_id0 =
        detail::AllocateStructureArray<rbmd::Id>(dihedral_count);
    full_data_->_h_dihedral_id1 =
        detail::AllocateStructureArray<rbmd::Id>(dihedral_count);
    full_data_->_h_dihedral_id2 =
        detail::AllocateStructureArray<rbmd::Id>(dihedral_count);
    full_data_->_h_dihedral_id3 =
        detail::AllocateStructureArray<rbmd::Id>(dihedral_count);
    for (std::size_t i = 0; i < dihedral_count; ++i) {
      const auto& dihedral = snapshot.dihedrals[i];
      full_data_->_h_dihedral_type[i] = dihedral.type;
      full_data_->_h_dihedral_id0[i] = dihedral.atom1;
      full_data_->_h_dihedral_id1[i] = dihedral.atom2;
      full_data_->_h_dihedral_id2[i] = dihedral.atom3;
      full_data_->_h_dihedral_id3[i] = dihedral.atom4;
    }
    const auto improper_count = snapshot.impropers.size();

    // Check if there are any impropers to process
    if (improper_count > 0) {
      // Allocate memory for each component of the improper data
      full_data_->_h_improper_type =
          detail::AllocateStructureArray<rbmd::Id>(improper_count);
      full_data_->_h_improper_id0 =
          detail::AllocateStructureArray<rbmd::Id>(improper_count);
      full_data_->_h_improper_id1 =
          detail::AllocateStructureArray<rbmd::Id>(improper_count);
      full_data_->_h_improper_id2 =
          detail::AllocateStructureArray<rbmd::Id>(improper_count);
      full_data_->_h_improper_id3 =
          detail::AllocateStructureArray<rbmd::Id>(improper_count);

      // Loop through each improper in the snapshot and copy its data
      for (std::size_t i = 0; i < improper_count; ++i) {
        const auto& improper = snapshot.impropers[i];
        full_data_->_h_improper_type[i] = improper.type;
        full_data_->_h_improper_id0[i] = improper.atom1;
        full_data_->_h_improper_id1[i] = improper.atom2;
        full_data_->_h_improper_id2[i] = improper.atom3;
        full_data_->_h_improper_id3[i] = improper.atom4;
      }
    }
  }

 private:
  std::shared_ptr<FullStructureData> full_data_;
};

inline std::unique_ptr<StructureAdapter> CreateStructureAdapter(
    const std::shared_ptr<StructureData>& data) {
  if (!data) {
    return nullptr;
  }

  if (auto full = std::dynamic_pointer_cast<FullStructureData>(data)) {
    return std::make_unique<FullStructureAdapter>(std::move(full));
  }
  if (auto charge = std::dynamic_pointer_cast<ChargeStructureData>(data)) {
    return std::make_unique<ChargeStructureAdapter>(std::move(charge));
  }
  if (auto basic = std::dynamic_pointer_cast<BasicStructureData>(data)) {
    return std::make_unique<BasicStructureAdapter>(std::move(basic));
  }
  if (auto atoms = std::dynamic_pointer_cast<AtomsStructureData>(data)) {
    return std::make_unique<AtomsStructureAdapter>(
        std::static_pointer_cast<StructureData>(atoms));
  }
  return std::make_unique<AtomsStructureAdapter>(data);
}

}  // namespace reader
