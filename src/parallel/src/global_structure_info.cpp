#include "mpi.h"
#include "common/rbmd_define.h"
#include "global_structure_info.h"


void BroadcastGlobalStructureInfo(GlobalStructureInfo& info, int root_rank,
                                     MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  MPI_Datatype box_type;
  const int box_nitems = 12; // Box 类中共有 12 个需要广播的成员变量块

  // 每个成员块的元素数量
  int box_blocklengths[box_nitems] = {1, 1, 1, 1, 6, 6, 3, 3, 3, 3, 3, 3};

  // 每个成员块的内存偏移量（位移）
  MPI_Aint box_displacements[box_nitems];

  // 每个成员块的数据类型
  // rbmd::Real 对应 MPI_RBMD_REAL, rbmd::Id 对应 MPI_RBMD_ID
  MPI_Datatype box_types[box_nitems] = {
      MPI_INT, // _type
      MPI_CXX_BOOL, // _pbc_x
      MPI_CXX_BOOL, // _pbc_y
      MPI_CXX_BOOL, // _pbc_z
      MPI_RBMD_REAL, // _length
      MPI_RBMD_REAL, // _length_inv
      MPI_RBMD_REAL, // _coord_min
      MPI_RBMD_REAL, // _coord_max
      MPI_RBMD_REAL, // _halo_coord_min
      MPI_RBMD_REAL, // _halo_coord_max
      MPI_RBMD_ID, // _box_width_as_cell_units
      MPI_RBMD_ID // _halo_width_as_cell_units
  };

  // 使用一个临时对象来获取真实的内存地址和偏移量
  Box temp_box;
  MPI_Aint base_address;
  MPI_Get_address(&temp_box, &base_address);

  // 依次获取每个成员的地址
  MPI_Get_address(&temp_box._type, &box_displacements[0]);
  MPI_Get_address(&temp_box._pbc_x, &box_displacements[1]);
  MPI_Get_address(&temp_box._pbc_y, &box_displacements[2]);
  MPI_Get_address(&temp_box._pbc_z, &box_displacements[3]);
  MPI_Get_address(&temp_box._length, &box_displacements[4]);
  MPI_Get_address(&temp_box._length_inv, &box_displacements[5]);
  MPI_Get_address(&temp_box._coord_min, &box_displacements[6]);
  MPI_Get_address(&temp_box._coord_max, &box_displacements[7]);
  MPI_Get_address(&temp_box._halo_coord_min, &box_displacements[8]);
  MPI_Get_address(&temp_box._halo_coord_max, &box_displacements[9]);
  MPI_Get_address(&temp_box._box_width_as_cell_units, &box_displacements[10]);
  MPI_Get_address(&temp_box._halo_width_as_cell_units, &box_displacements[11]);

  // 计算相对于基地址的偏移量
  for (int i = 0; i < box_nitems; ++i) {
    box_displacements[i] -= base_address;
  }

  // 创建并提交最终的 Box MPI 数据类型
  MPI_Type_create_struct(box_nitems, box_blocklengths, box_displacements,
                         box_types, &box_type);
  MPI_Type_commit(&box_type);

  // --- 2. 创建 GlobalStructureInfo 的 MPI 数据类型 ---
  MPI_Datatype global_info_type;
  const int global_nitems = 10; // GlobalStructureInfo 中有10个成员
  int global_blocklengths[global_nitems] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  MPI_Aint global_displacements[global_nitems];
  MPI_Datatype global_types[global_nitems] = {
      MPI_RBMD_ID, MPI_RBMD_ID, MPI_RBMD_ID, MPI_RBMD_ID, MPI_RBMD_ID,
      MPI_RBMD_ID, MPI_RBMD_ID, MPI_RBMD_ID, MPI_RBMD_ID,
      box_type // 使用我们刚刚创建的 box_type
  };

  // 获取偏移量
  GlobalStructureInfo temp_info;
  MPI_Aint base_address_2;
  MPI_Get_address(&temp_info, &base_address_2);
  MPI_Get_address(&temp_info.total_atoms, &global_displacements[0]);
  MPI_Get_address(&temp_info.total_bonds, &global_displacements[1]);
  MPI_Get_address(&temp_info.total_angles, &global_displacements[2]);
  MPI_Get_address(&temp_info.total_dihedrals, &global_displacements[3]);
  MPI_Get_address(&temp_info.total_impropers, &global_displacements[4]);
  MPI_Get_address(&temp_info.num_atom_types, &global_displacements[5]);
  MPI_Get_address(&temp_info.num_bond_types, &global_displacements[6]);
  MPI_Get_address(&temp_info.num_angle_types, &global_displacements[7]);
  MPI_Get_address(&temp_info.num_dihedral_types, &global_displacements[8]);
  MPI_Get_address(&temp_info.global_box, &global_displacements[9]);

  // 计算相对偏移量
  for (int i = 0; i < global_nitems; ++i) {
    global_displacements[i] -= base_address_2;
  }

  // 创建并提交 GlobalStructureInfo 的 MPI 类型
  MPI_Type_create_struct(global_nitems, global_blocklengths,
                         global_displacements, global_types, &global_info_type);
  MPI_Type_commit(&global_info_type);

  // --- 3. 广播 ---
  // 假设 rank 0 是根进程，它已经初始化了 info 结构体
  MPI_Bcast(&info, 1, global_info_type, root_rank, comm);

  // --- 4. 释放资源 ---
  MPI_Type_free(&box_type);
  MPI_Type_free(&global_info_type);

  // 现在，所有进程的 info 结构体都已更新
}