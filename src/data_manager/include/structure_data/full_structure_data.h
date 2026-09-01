#pragma once
#include "basic_structure_data.h"
#include <vector>             //
#include <utility>            //

class FullStructureData : public BasicStructureData {
 public:
  bool checkStructure() const override {
    if (false == BasicStructureData::checkStructure()) {
      return false;
    }

    //
    return true;
  }

  rbmd::Id* _h_molecules_id = nullptr;
  /// bond
  rbmd::Id* _h_bond_type = nullptr;
  rbmd::Id* _h_bond_id0 = nullptr;
  rbmd::Id* _h_bond_id1 = nullptr;
  std::vector<std::pair<rbmd::Id, rbmd::Id>> special_pairs_12;

  // per-atom bond topology
  int* _h_num_bond = nullptr;
  int* _h_bond_type_per_atom = nullptr;
  rbmd::Id* _h_bond_atom_per_atom = nullptr;
  
  //special
  rbmd::Real* _h_special_weights = nullptr;
  rbmd::Id* _h_special_ids = nullptr;
  rbmd::Id* _h_special_offsets = nullptr;
  rbmd::Id* _h_special_offset_count = nullptr;

  rbmd::Id*  _h_atoms_vec_gro = nullptr;
  rbmd::Id*  _h_count_vector = nullptr;
  rbmd::Id*  _h_atoms_offset = nullptr;

  //
  rbmd::Id _num_special_weights = 0;
  rbmd::Id _num_special_ids = 0;
  rbmd::Id _num_special_offsets = 0;
  rbmd::Id _num_special_offset_count = 0;

  rbmd::Id _num_atoms_vec_gro = 0;
  rbmd::Id _num_count_vector = 0;
  rbmd::Id _num_atoms_offset = 0;

  /// angle
  rbmd::Id* _h_angle_type = nullptr;
  rbmd::Id* _h_angle_id0 = nullptr;
  rbmd::Id* _h_angle_id1 = nullptr;
  rbmd::Id* _h_angle_id2 = nullptr;
  int3* _h_angle_id_vec = nullptr;
 std::vector<std::pair<rbmd::Id, rbmd::Id>> special_pairs_13;

  // per-atom angle topology
  int* _h_num_angle = nullptr;
  int* _h_angle_type_per_atom = nullptr;
  rbmd::Id* _h_angle_atom1_per_atom = nullptr;
  rbmd::Id* _h_angle_atom2_per_atom = nullptr;
  rbmd::Id* _h_angle_atom3_per_atom = nullptr;

  /// dihedral
  rbmd::Id* _h_dihedral_type = nullptr;
  rbmd::Id* _h_dihedral_id0 = nullptr;
  rbmd::Id* _h_dihedral_id1 = nullptr;
  rbmd::Id* _h_dihedral_id2 = nullptr;
  rbmd::Id* _h_dihedral_id3 = nullptr;
   std::vector<std::pair<rbmd::Id, rbmd::Id>>  special_pairs_14;

  // per-atom dihedral topology
  int* _h_num_dihedral = nullptr;
  int* _h_dihedral_type_per_atom = nullptr;
  rbmd::Id* _h_dihedral_atom1_per_atom = nullptr;
  rbmd::Id* _h_dihedral_atom2_per_atom = nullptr;
  rbmd::Id* _h_dihedral_atom3_per_atom = nullptr;
  rbmd::Id* _h_dihedral_atom4_per_atom = nullptr;

  /// improper
  rbmd::Id* _h_improper_type = nullptr;
  rbmd::Id* _h_improper_id0 = nullptr;
  rbmd::Id* _h_improper_id1 = nullptr;
  rbmd::Id* _h_improper_id2 = nullptr;
  rbmd::Id* _h_improper_id3 = nullptr;

  // per-atom improper topology
  int* _h_num_improper = nullptr;
  int* _h_improper_type_per_atom = nullptr;
  rbmd::Id* _h_improper_atom1_per_atom = nullptr;
  rbmd::Id* _h_improper_atom2_per_atom = nullptr;
  rbmd::Id* _h_improper_atom3_per_atom = nullptr;
  rbmd::Id* _h_improper_atom4_per_atom = nullptr;

  //rbmd::Id* _h_special_source_array;
  //rbmd::Id* _h_special_offsets_array;

  // special bond topology (LAMMPS nspecial/special layout)
  int* _h_nspecial = nullptr;     // [nmax * 3]
  rbmd::Id* _h_special = nullptr; // [nmax * maxspecial]
  int _h_maxspecial = 0;
  rbmd::Id _h_nmax_special = 0;
};
