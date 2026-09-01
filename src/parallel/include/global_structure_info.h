
#pragma once
#include "../common/types.h"
#include "model/box.h"
#include "mpi.h"


struct GlobalStructureInfo {
  rbmd::Id total_atoms;      
  rbmd::Id total_bonds;      
  rbmd::Id total_angles;     
  rbmd::Id total_dihedrals;  
  rbmd::Id total_impropers;  

  rbmd::Id num_atom_types;     
  rbmd::Id num_bond_types;     
  rbmd::Id num_angle_types;    
  rbmd::Id num_dihedral_types;

  Box global_box;
};


void BroadcastGlobalStructureInfo(GlobalStructureInfo& info, int root_rank,
                                     MPI_Comm comm);