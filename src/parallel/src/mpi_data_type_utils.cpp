#include "mpi_data_type_utils.h"
// 获取 _leaving_atom_type，如果没有缓存，则创建并缓存
MPI_Datatype MpiDataTypeUtils::GetLeavingAtomType() {
  if (_leaving_atom_type == MPI_DATATYPE_NULL) {
    _leaving_atom_type = CreateLeavingAtomMpiType();
  }
  return _leaving_atom_type;
}

// 获取 _halo_atom_type，如果没有缓存，则创建并缓存
MPI_Datatype MpiDataTypeUtils::GetHaloAtomType() {
  if (_halo_atom_type == MPI_DATATYPE_NULL) {
    _halo_atom_type = CreateHaloAtomMpiType();
  }
  return _halo_atom_type;
}

// 创建 LeavingAtom 对应的 MPI 数据类型
MPI_Datatype MpiDataTypeUtils::CreateLeavingAtomMpiType() {
  MPI_Datatype types[8] = {
    MPI_RBMD_ID,   // id
    MPI_RBMD_REAL, // rx
    MPI_RBMD_REAL, // ry
    MPI_RBMD_REAL, // rz
    MPI_RBMD_REAL, // vx
    MPI_RBMD_REAL, // vy
    MPI_RBMD_REAL, // vz
    MPI_RBMD_REAL  // charge
  };

  int blockLengths[8] = {1, 1, 1, 1, 1, 1, 1, 1};  // 每个成员1个单位
  MPI_Aint displacements[8];

  // 偏移量计算
  displacements[0] = 0;
  displacements[1] = sizeof(rbmd::Id);
  displacements[2] = displacements[1] + sizeof(rbmd::Real);
  displacements[3] = displacements[2] + sizeof(rbmd::Real);
  displacements[4] = displacements[3] + sizeof(rbmd::Real);
  displacements[5] = displacements[4] + sizeof(rbmd::Real);
  displacements[6] = displacements[5] + sizeof(rbmd::Real);
  displacements[7] = displacements[6] + sizeof(rbmd::Real);

  MPI_Type_create_struct(8, blockLengths, displacements, types, &this->_leaving_atom_type);
  MPI_Type_commit(&this->_leaving_atom_type);

  return this->_leaving_atom_type;
}

// 创建 HaloAtom 对应的 MPI 数据类型
MPI_Datatype MpiDataTypeUtils::CreateHaloAtomMpiType() {
  MPI_Datatype types[4] = {
    MPI_RBMD_REAL, // rx
    MPI_RBMD_REAL, // ry
    MPI_RBMD_REAL, // rz
    MPI_RBMD_REAL  // charge
  };

  int blockLengths[4] = {1, 1, 1, 1};  // 每个成员1个单位
  MPI_Aint displacements[4];

  // 偏移量计算
  displacements[0] = 0;
  displacements[1] = sizeof(rbmd::Real);
  displacements[2] = displacements[1] + sizeof(rbmd::Real);
  displacements[3] = displacements[2] + sizeof(rbmd::Real);

  MPI_Type_create_struct(4, blockLengths, displacements, types, &this->_halo_atom_type);
  MPI_Type_commit(&this->_halo_atom_type);

  return this->_halo_atom_type;
}

